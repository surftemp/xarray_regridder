#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

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
import pyproj
import numpy as np
import sys

from xarray_regridder import VERSION as XARRAY_REGRIDDER_VERSION

class Progress(object):

    def __init__(self,label):
        self.label = label
        self.last_progress_frac = None

    def report(self,msg,progress_frac):
        if self.last_progress_frac == None or (progress_frac - self.last_progress_frac) >= 0.01:
            self.last_progress_frac = progress_frac
            i = int(100*progress_frac)
            if i > 100:
                i = 100
            si = i // 2
            sys.stdout.write("\r%s %s %-05s %s" % (self.label,msg,str(i)+"%","#"*si))
            sys.stdout.flush()

    def complete(self,msg):
        self.report(msg, 1)
        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser(prog='nearest_s2d_regridder', usage='%(prog)s [options]')
    parser.add_argument('-V', '--version', action='version', version="%(prog)s " + XARRAY_REGRIDDER_VERSION)

    parser.add_argument("input_path",
                        help="input data file to be regridded or folder containing .nc files to be regridded")
    parser.add_argument("target_grid_path",
                        help="path to file defining grid with 1D lat/lon onto which data is to be regridded")
    parser.add_argument("output_path",
                        help="path to file to hold combined regridded data OR folder to hold individually regridded files")

    parser.add_argument("--target-y", type=str, help="target grid y coordinate variable name", default="x")
    parser.add_argument("--target-x", type=str, help="target grid x coordinate variable name", default="y")
    parser.add_argument("--target-crs", type=int, help="target CRS, given as an EPSG number", default=27700)

    parser.add_argument("--source-y", type=str, help="input y coordinate variable name", default="latitude")
    parser.add_argument("--source-x", type=str, help="input x coordinate variable name", default="longitude")
    parser.add_argument("--source-crs", type=int, help="source CRS, given as an EPSG number", default=4326)

    parser.add_argument("--variables", nargs="+", help="Specify variables to process", default=["AOD_0550","AOD_0659"])
    parser.add_argument("--distance-limit", type=float, help="Specify distance limit in target grid units", default=3300)
    parser.add_argument("--window-size", type=float, help="Specify size of sliding window on target grid, in pixels", default=1000)
    parser.add_argument("--window-offset", type=float, help="Specify number of pixels to slide window on each step", default=950)
    parser.add_argument("--limit", type=int, help="Useful for debugging, process only this many windows", default=0)
    parser.add_argument("--fill-value", type=float, help="Set the _FillValue to this number", default=None)

    parser.add_argument("--chuk", action="store_true", help="Perform CHUK specific optimisations")
    parser.add_argument(
        "--check-version",
        help="check that the version number of this tool matches the specified version string",
        default=None
    )

    args = parser.parse_args()

    if args.check_version is not None:
        if XARRAY_REGRIDDER_VERSION != args.check_version:
            print(f"Version of this tool {XARRAY_REGRIDDER_VERSION} does not match requested version {args.check_version}")
            sys.exit(-1)

    # get the coordinates of each input data pixel
    input_ds = xr.open_dataset(args.input_path)

    if args.chuk:
        # define a conservative lat-lon bounding box for the CHUK grid
        chuk_lat_min = 47
        chuk_lat_max = 62
        chuk_lon_min = -16
        chuk_lon_max = 5

        # get the lat-lon bounding box of the input data
        input_lat_min = input_ds[args.source_y].min().item()
        input_lat_max = input_ds[args.source_y].max().item()
        input_lon_min = input_ds[args.source_x].min().item()
        input_lon_max = input_ds[args.source_x].max().item()

        # if there can be no intersection between the input and the chuk grid, stop now
        if input_lat_min > chuk_lat_max or \
            input_lat_max < chuk_lat_min or \
            input_lon_min > chuk_lon_max or \
            input_lon_max < chuk_lon_min:
            print(f"No intersection - skipping writing empty output to {args.output_path}")
            return

        # null out any source coordinates which must lie outside the CHUK area
        # this is to guard against unstable projection behaviour noticed by KJP
        source_y = input_ds[args.source_y].data
        source_x = input_ds[args.source_x].data

        cond = np.logical_and(
            np.logical_and(source_y <= chuk_lat_max, source_y >= chuk_lat_min),
            np.logical_and(source_x <= chuk_lon_max, source_y >= chuk_lon_min))

        input_ds[args.source_y].data = np.where(cond, source_y, np.nan)
        input_ds[args.source_x].data = np.where(cond, source_x, np.nan)

    source_y = input_ds[args.source_y].data.flatten()
    source_x = input_ds[args.source_x].data.flatten()

    # map the coordinates to the target CRS
    # note - where input coordinates are nan, the resulting output x,y values are set to inf
    transformer = pyproj.Transformer.from_crs(args.source_crs, args.target_crs, always_xy=True)
    x, y = transformer.transform(source_x, source_y)

    # create a representation of the input data in flattened form
    ds2 = xr.Dataset()
    ds2["x"] = xr.DataArray(x, dims=("case",))
    ds2["y"] = xr.DataArray(y, dims=("case",))
    for v in args.variables:
        ds2[v] = xr.DataArray(input_ds[v].data.flatten(), dims=("case",))

    # create the output dataset based on the target grid
    grid_ds = xr.open_dataset(args.target_grid_path)

    output_ds = xr.Dataset(attrs=input_ds.attrs)

    output_ds["x"] = grid_ds.x
    output_ds["x_bnds"] = grid_ds.x_bnds
    output_ds["y"] = grid_ds.y
    output_ds["y_bnds"] = grid_ds.y_bnds

    # setup the windowing parameters
    window_size = args.window_size
    offset = args.window_offset
    distance_limit = args.distance_limit

    w = output_ds["x"].shape[0]
    h = output_ds["y"].shape[0]

    # create requested variables in the output dataset
    for v in args.variables:
        output_ds[v] = xr.DataArray(np.zeros((h,w)),dims=("y","x"),attrs=input_ds[v].attrs)
        output_ds[v][::] = np.nan

    # create an array to track the distance to centre of each filled in pixel
    output_ds["distances"] = xr.DataArray(np.zeros((h,w)),dims=("y","x"))
    output_ds["distances"][::] = np.nan

    # define the area to process, by default the whole of the target grid
    start_y = 0
    end_y = h
    start_x = 0
    end_x = w

    # divide the area into windows and process each window.
    # this should reduce the computation time

    # work out how many windows
    total_windows = args.limit
    if not total_windows:
        for y_off in range(start_y, end_y, offset):
            for x_off in range(start_x, end_x, offset):
                total_windows += 1

    window_count = 0

    # track how many pixels are copied in
    total_n = 0

    p = Progress("Processed")
    for y_off in range(start_y, end_y, offset):

        if args.limit and window_count >= args.limit:
            break

        for x_off in range(start_x, end_x, offset):

            if args.limit and window_count >= args.limit:
                break

            # process a single window
            local_data = {}
            for v in args.variables:
                local_data[v] = output_ds[v][y_off:min(y_off+window_size,end_y),x_off:min(x_off+window_size,end_x)]
            local_distances = output_ds["distances"][y_off:min(y_off+window_size,end_y),x_off:min(x_off+window_size,end_x)]

            lh = local_distances.shape[0]
            lw = local_distances.shape[1]

            # get the bounds
            min_x = local_distances.x.min()
            max_x = local_distances.x.max()
            min_y = local_distances.y.min()
            max_y = local_distances.y.max()

            # work out which of the source pixels fall into this window
            include = np.logical_and(np.logical_and(ds2.x >= min_x, ds2.x <= max_x),
                                     np.logical_and(ds2.y >= min_y, ds2.y <= max_y))

            ds2_filtered = ds2.where(include,drop=True)

            n_filtered = ds2_filtered.x.shape[0]
            total_n += n_filtered

            if n_filtered == 0:
                continue

            # expand x and y to 2-d arrays for the distance calculation
            local_x = local_distances.x.expand_dims({"y":lh},axis=0).astype(float)
            local_y = local_distances.y.expand_dims({"x":lw},axis=1).astype(float)

            # work through each of the input pixels...
            for i in range(n_filtered):

                vx = ds2_filtered.x[i].item()
                vy = ds2_filtered.y[i].item()

                # get the squared distance from each window pixel to the input pixel
                v_sq_distances = (local_x - vx)**2 + (local_y - vy)**2

                # work out which window pixels to update
                update = np.logical_and(v_sq_distances.data < (distance_limit**2),
                                np.logical_or(
                                    np.isnan(local_distances.data),
                                    v_sq_distances.data<local_distances.data))

                # apply the updates to the window arrays for each variable
                for v in args.variables:
                    value = ds2_filtered[v][i].item()
                    local_data[v].data = np.where(update.data,value,local_data[v].data)
                # track the distances
                local_distances.data = xr.where(update.data, v_sq_distances.data, local_distances.data)

            # write the results for this window back to the output dataset
            for v in args.variables:
                output_ds[v][y_off:min(y_off + window_size, end_y), x_off:min(x_off + window_size, end_x)] = local_data[v]

            output_ds["distances"][y_off:min(y_off + window_size, end_y),
                              x_off:min(x_off + window_size, end_x)] = local_distances

            window_count += 1
            p.report(f"{n_filtered} pixels in area ({y_off},{x_off})", window_count/total_windows)

    # if no pixels are included, don't write an empty output file
    if total_n == 0:
        print(f"Skipping writing empty output to {args.output_path}")
        return

    # remove the distances variable unless needed for debugging
    del output_ds["distances"]

    encoding = {}
    for v in args.variables:
        encoding[v] = { "zlib": True, "complevel": 5, "dtype": np.float32}
        if args.chuk:
            encoding[v]["chunksizes"] = [1000, 1000]
        if args.fill_value is not None:
            encoding[v]["_FillValue"] = args.fill_value

    output_ds.to_netcdf(args.output_path, encoding=encoding)

if __name__ == '__main__':
    main()