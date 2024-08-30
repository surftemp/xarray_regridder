#!/usr/bin/env python
# -*- coding: utf-8 -*-

#     EOCIS high resolution data processing for the British Isles
#
#     Copyright (C) 2023-2024  EOCIS and National Centre for Earth Observation (NCEO)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import xarray as xr
import numpy as np
import os
import logging
import sys
import pyproj
import time
import enum

RETRY_DELAY = 60

class DistanceMethods(enum.Enum):

    CARTESIAN = 1   # calculate distance between pairs of (x,y) points on a cartesian plane
    SPHERICAL = 2   # calculate approximate distance between pairs of (lat,lon) points on a sphere

class Regridder:

    def __init__(self, variables, grid_ds,
                 source_x, source_y, source_crs, target_x, target_y, target_crs):
        """
        Create a regridder

        :param variables: list of variable input_name:mode:output_name from the source dataset(s) to regrid
        :param grid_ds: the target grid dataset
        :param source_x: the name of the x-coordinate in the source dataset(s)
        :param source_y: the name of the y-coordinate in the source dataset(s)
        :param source_crs: the source CRS (as an EPSG number)
        :param target_x: the name of the x-coordinate in the target grid
        :param target_y: the name of the y-coordinate in the target grid
        :param target_crs: the target CRS (as an EPSG number)
        """
        self.variables = [self.decode_variable_mode(v) for v in variables]
        self.source_x = source_x
        self.source_y = source_y
        self.source_crs = source_crs
        self.target_x = target_x
        self.target_y = target_y
        self.target_crs = target_crs

        self.target_height = None
        self.target_width = None
        self.target_x_dim = None
        self.target_y_dim = None

        self.dataset_attrs = {}
        self.variable_attrs = {}

        self.dtypes = {}

        self.logger = logging.getLogger("Regrid")

        output_names = set()

        self.input_vars = set()
        for (v,mode,output_name) in self.variables:
            self.input_vars.add(v)
            if output_name in output_names:
                raise ValueError(f"output variables have duplicate names: {output_name}")
            else:
                output_names.add(output_name)

        # work out whether distances between source and destination pixels are needed, and which method to use
        self.compute_distances = False
        self.distance_method = None
        for (v, mode, output_name) in self.variables:
            if mode == "nearest" or mode == "distance":
                self.compute_distances = True
                if target_crs == 4326:
                    self.distance_method = DistanceMethods.SPHERICAL
                else:
                    crs = pyproj.CRS(f"EPSG:{self.target_crs}")
                    if crs.axis_info[0].unit_name == 'metre':
                        self.distance_method = DistanceMethods.CARTESIAN
                    else:
                        raise ValueError("Nearest neighbour currently only supported if the target CRS is EPSG:4326 or has metre as its unit")

        self.grid = grid_ds

        # check target x and y are 1 dimensional
        if len(self.grid[self.target_x].shape) != 1:
            self.logger.error("target grid x variable should have only 1 dimension")
            sys.exit(-1)

        if len(self.grid[self.target_y].shape) != 1:
            self.logger.error("target grid y variable should have only 1 dimension")
            sys.exit(-1)

        # work out the indices on the target grid
        self.target_y0 = float(self.grid[self.target_y][0])
        self.target_yN = float(self.grid[self.target_y][-1])
        self.target_x0 = float(self.grid[self.target_x][0])
        self.target_xN = float(self.grid[self.target_x][-1])

        # get the target grid dimensions and size
        self.target_height = self.grid[self.target_y].shape[0]
        self.target_width = self.grid[self.target_x].shape[0]

        self.target_x_dim = self.grid[self.target_x].dims[0]
        self.target_y_dim = self.grid[self.target_y].dims[0]

        self.reset()

    def decode_variable_mode(self, variable):
        splits = variable.split(":")
        output_name = ""
        if len(splits) == 3:
            variable_name = splits[0]
            mode = splits[1]
            output_name = splits[2]
        elif len(splits) == 2:
            variable_name = splits[0]
            mode = splits[1]
        elif len(splits) == 1:
            variable_name = variable
            mode = "nearest"
        else:
            raise ValueError(
                f"Invalid variable directive: {variable} should be INPUT_NAME or INPUT_NAME:MODE or INPUT_NAME:MODE:OUTPUT_NAME")
        if mode not in ["min", "max", "mean", "nearest", "count", "sum", "distance"]:
            raise ValueError(f"Invalid mode {mode} for variable {variable}")
        if output_name == "":
            output_name = variable_name + "_" + mode
        return variable_name, mode, output_name

    def ingest(self,ds, stride=1):

        for v in self.input_vars:
            da = ds[v].squeeze()
            if len(da.dims) != 2:
                self.logger.error(
                    f"variable {v} does not have exactly two non-unit dimensions, ignoring this input data")
                return False

            if v not in self.variable_attrs:
                self.variable_attrs[v] = da.attrs
            if v not in self.dtypes:
                self.dtypes[v] = da.dtype

        self.dataset_attrs = ds.attrs

        if len(ds[self.source_y].shape) == 1:
            shape = (len(ds[self.source_y]), len(ds[self.source_x]))
            source_y_dim = ds[self.source_y].dims[0]
            source_x_dim = ds[self.source_x].dims[0]
            y2d = xr.DataArray(np.broadcast_to(ds[self.source_y].data[None].T, shape),
                               dims=(source_y_dim, source_x_dim))
            x2d = xr.DataArray(np.broadcast_to(ds[self.source_x], shape), dims=(source_y_dim, source_x_dim))
        else:
            shape = ds[self.source_y].shape
            x2d = ds[self.source_x]
            y2d = ds[self.source_y]
            source_y_dim = ds[self.source_x].dims[0]
            source_x_dim = ds[self.source_x].dims[1]

        (source_height, source_width) = shape

        target_shape = (len(self.grid[self.target_y]), len(self.grid[self.target_x]))
        target_y2d = np.broadcast_to(self.grid[self.target_y].data[None].T, target_shape)
        target_x2d = np.broadcast_to(self.grid[self.target_x], target_shape)

        if self.source_crs != self.target_crs:
            transformer = pyproj.Transformer.from_crs(self.source_crs, self.target_crs)
            x2d, y2d = transformer.transform(y2d.data, x2d.data)
            y2d = xr.DataArray(y2d, dims=(source_y_dim, source_x_dim))
            x2d = xr.DataArray(x2d, dims=(source_y_dim, source_x_dim))

        indices_nj = np.int32(np.round((self.target_height - 1) * (y2d.data - self.target_y0) / (self.target_yN - self.target_y0)))
        indices_ni = np.int32(np.round((self.target_width - 1) * (x2d.data - self.target_x0) / (self.target_xN - self.target_x0)))

        # set indices to (target_height,target_width) for points that lie outside the target grid
        indices_nj = np.where(indices_nj < 0, self.target_height, indices_nj)
        indices_nj = np.where(indices_nj >= self.target_height, self.target_height, indices_nj)
        indices_ni = np.where(indices_ni < 0, self.target_width, indices_ni)
        indices_ni = np.where(indices_ni >= self.target_width, self.target_width, indices_ni)

        indices_nj = xr.DataArray(indices_nj, dims=(source_y_dim, source_x_dim))
        indices_ni = xr.DataArray(indices_ni, dims=(source_y_dim, source_x_dim))

        count = np.where(np.logical_or(indices_ni == self.target_width, indices_nj == self.target_height), 0, 1).sum()
        self.logger.info(f"Found {count} intersecting pixels")

        if count == 0:
            self.logger.warning("No intersecting pixels, skipping")
            return False

        class InvalidStrideException(Exception):
            pass

        # establish the stride - so that the number of passes over the input data is minimised but
        # ensures that no input pixels are assigned to the same output pixel on a pass

        while True:
            indices_by_slice = {}
            try:
                for xs in range(0, stride):
                    for ys in range(0, stride):
                        s = {}
                        s[source_x_dim] = slice(xs, None, stride)
                        s[source_y_dim] = slice(ys, None, stride)
                        iy = indices_nj.isel(**s)
                        ix = indices_ni.isel(**s)
                        indices_by_slice[(ys, xs)] = (s, iy.data, ix.data)

                        # check that in each stride, the set of valid target indices are unique
                        # if this is not the case, some source values will be ignored, raise an exception
                        ones = xr.DataArray(np.ones((source_height,source_width),dtype=int),dims=(self.source_y,self.source_x))
                        target_data = xr.DataArray(np.zeros((self.target_height+1,self.target_width+1),dtype=int),dims=(self.target_y,self.target_x))
                        target_data[iy, ix] = ones.isel(**s).data
                        valid_target_data = target_data[:-1, :-1]
                        valid_target_count = valid_target_data.sum().item()
                        expected_target_count = xr.where(np.logical_and(ix < self.target_width, iy < self.target_height),1,0).sum().item()
                        if valid_target_count != expected_target_count:
                            raise InvalidStrideException()
                break
            except InvalidStrideException:
                stride *= 2
                self.logger.warning(f"Increasing stride to {stride} to avoid data loss")

        for xs in range(0, stride):
            for ys in range(0, stride):
                slice_idx = xs * stride + ys
                self.logger.info(f"\t\tProcessing slice {slice_idx + 1}/{stride ** 2}")
                s, iy, ix = indices_by_slice[(ys, xs)]
                sq_dist = None
                if self.compute_distances:
                    # calculate squared distances for this stride and cache them if not already cached
                    source_coords_x = np.zeros((self.target_height + 1, self.target_width + 1))
                    source_coords_x[:, :] = np.nan
                    source_coords_y = np.zeros((self.target_height + 1, self.target_width + 1))
                    source_coords_y[:, :] = np.nan
                    source_coords_x[iy, ix] = x2d.isel(**s).data
                    source_coords_y[iy, ix] = y2d.isel(**s).data
                    if self.distance_method is DistanceMethods.CARTESIAN:
                        sq_dist = np.power(source_coords_y[:-1, :-1] - target_y2d, 2) \
                                             + np.power(source_coords_x[:-1, :-1] - target_x2d, 2)
                    else:
                        # x is a longitude, adjust using cos(radians(y))
                        sq_dist = np.power(
                            np.cos(np.radians(target_y2d)) * source_coords_x[:-1, :-1] - target_x2d, 2) \
                                             + np.power(source_coords_y[:-1, :-1] - target_y2d, 2)

                for v in self.input_vars:
                    self.logger.info(f"\t\tCalculating statistics for {v}")

                    da = ds[v].squeeze()

                    target_data = np.zeros((self.target_height + 1, self.target_width + 1))
                    target_data[:, :] = np.nan

                    target_data[iy, ix] = da.isel(**s).data
                    valid_target_data = target_data[:-1, :-1]

                    # accumulate statistics for this variable on this stride
                    if v in self.accumulated_mins:
                        self.accumulated_mins[v] = np.fmin(valid_target_data, self.accumulated_mins[v])

                    if v in self.accumulated_maxes:
                        self.accumulated_maxes[v] = np.fmax(valid_target_data, self.accumulated_maxes[v])

                    if v in self.accumulated_counts:
                        self.accumulated_counts[v] = self.accumulated_counts[v] + np.where(np.isnan(valid_target_data), 0, 1)

                    if v in self.accumulated_sums:
                        self.accumulated_sums[v] = self.accumulated_sums[v] \
                                                    + np.where(np.isnan(valid_target_data), 0, valid_target_data)

                    if v in self.accumulated_nearest:
                        self.accumulated_nearest[v] = \
                            np.where(np.logical_or(np.isnan(sq_dist),np.isnan(valid_target_data)),
                                self.accumulated_nearest[v],
                                np.where(sq_dist < self.accumulated_sqdistances[v],
                                    valid_target_data,
                                    self.accumulated_nearest[v]))

                    if v in self.accumulated_sqdistances:
                        self.accumulated_sqdistances[v] = \
                            np.where(np.logical_or(np.isnan(sq_dist),np.isnan(valid_target_data)),
                                self.accumulated_sqdistances[v],
                                np.where(sq_dist < self.accumulated_sqdistances[v],
                                    sq_dist,
                                    self.accumulated_sqdistances[v]))


        return True

    def get_output(self, time_da=None, chunk_sizes=None):

        output_ds = self.grid.copy()
        if time_da is not None:
            output_ds["time"] = time_da
            dims = ("time", self.target_y_dim, self.target_x_dim)
        else:
            dims = (self.target_y_dim, self.target_x_dim)
        encodings = {}

        for (v,mode,output_variable) in self.variables:

            encodings[output_variable] = {"zlib": True, "complevel": 5, "dtype": str(self.dtypes[v])}

            accumulated = None
            if mode == "mean":
                accumulated = np.where(self.accumulated_counts[v] > 0, self.accumulated_sums[v]/self.accumulated_counts[v], np.nan)
            elif mode == "max":
                accumulated = self.accumulated_maxes[v]
            elif mode == "min":
                accumulated = self.accumulated_mins[v]
            elif mode == "nearest":
                accumulated = self.accumulated_nearest[v]
            elif mode == "count":
                accumulated = self.accumulated_counts[v]
                encodings[output_variable] = {"zlib": True, "complevel": 5, "dtype": "int32", "_FillValue": -999}
            elif mode == "sum":
                accumulated = self.accumulated_sums[v]
            elif mode == "distance":
                accumulated = np.sqrt(np.where(self.accumulated_sqdistances[v] < np.finfo(np.float32).max, self.accumulated_sqdistances[v], np.nan))

            if chunk_sizes:
                encodings[output_variable]["chunksizes"] = chunk_sizes

            if time_da is not None:
                accumulated = np.expand_dims(accumulated,axis=0)

            if np.issubdtype(self.dtypes[v], np.integer):
                encodings[output_variable]["_FillValue"] = -999

            da = xr.DataArray(data=accumulated,dims=dims,attrs=self.variable_attrs.get(v,None))
            output_ds[output_variable] = da

        for (name,value) in self.dataset_attrs.items():
            output_ds.attrs[name] = value

        output_ds.set_coords([self.target_y, self.target_x])
        return output_ds, encodings

    def reset(self):
        self.dtypes = {}
        self.dataset_attrs = {}
        self.variable_attrs = {}

        # (re)create dictionaries that map from a variable name to a numpy array on the target grid
        # which accumulates statistics on that variable
        self.accumulated_sums = {}        # variable name => sum of source pixels
        self.accumulated_counts = {}      # variable name => count of source pixels
        self.accumulated_mins = {}        # variable name => min of source pixels
        self.accumulated_maxes = {}       # variable name => max of source pixels
        self.accumulated_nearest = {}     # variable name => value of "nearest" source pixel
        self.accumulated_sqdistances = {} # variable name => squared distance to nearest valid value

        for (v,mode,output_name) in self.variables:
            if mode == "mean":
                if v not in self.accumulated_counts:
                    self.accumulated_counts[v] = np.zeros((self.target_height, self.target_width))
                if v not in self.accumulated_sums:
                    self.accumulated_sums[v] = np.zeros((self.target_height, self.target_width))
            elif mode == "min":
                if v not in self.accumulated_mins:
                    a = np.zeros((self.target_height, self.target_width))
                    a[:, :] = np.nan
                    self.accumulated_mins[v] = a
            elif mode == "max":
                if v not in self.accumulated_maxes:
                    a = np.zeros((self.target_height, self.target_width))
                    a[:, :] = np.nan
                    self.accumulated_maxes[v] = a
            elif mode == "nearest":
                if v not in self.accumulated_nearest:
                    a = np.zeros((self.target_height, self.target_width))
                    a[:, :] = np.nan
                    self.accumulated_nearest[v] = a
                if v not in self.accumulated_sqdistances:
                    self.accumulated_sqdistances[v] = np.zeros((self.target_height, self.target_width), dtype="float32")
                    self.accumulated_sqdistances[v][:, :] = np.finfo(np.float32).max
            elif mode == "count":
                if v not in self.accumulated_counts:
                    self.accumulated_counts[v] = np.zeros((self.target_height, self.target_width))
            elif mode == "sum":
                if v not in self.accumulated_sums:
                    self.accumulated_sums[v] = np.zeros((self.target_height, self.target_width))
            elif mode == "distance":
                if v not in self.accumulated_sqdistances:
                    self.accumulated_sqdistances[v] = np.zeros((self.target_height, self.target_width), dtype="float32")
                    self.accumulated_sqdistances[v][:, :] = np.finfo(np.float32).max

