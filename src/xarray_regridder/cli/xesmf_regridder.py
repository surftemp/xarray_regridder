#!/usr/bin/env python
# -*- coding: utf-8 -*-

#     EOCIS high resolution data processing for the British Isles
#
#     Copyright (C) 2023  EOCIS and National Centre for Earth Observation (NCEO)
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
import xesmf as xe
import numpy as np
import os
import json

m_per_degree = 111000

def copy_metadata_ds(ds, ds_ref):
    # maybe there is a quicker way to do this

    # copy over global attributes
    for (name, value) in ds_ref.attrs.items():
        if name not in ds.attrs:
            ds.attrs[name] = value

    # copy over attributes for each variable
    for (vname, v) in ds_ref.variables.items():
        if vname in ds.variables:
            for (name, value) in v.attrs.items():
                if name not in ds[vname].attrs:
                    ds[vname].attrs[name] = value

def copy_metadata(input_path, reference_path, output_path):

    ds = xr.open_dataset(input_path)
    ds_ref = xr.open_dataset(reference_path)

    copy_metadata(ds, ds_ref)

    ds.to_netcdf(output_path)

def write_cache(cache_folder, regridder, fingerprint):
    os.makedirs(cache_folder, exist_ok=True)
    filecount = len(os.listdir(cache_folder))
    subfolder = os.path.join(cache_folder, f"regridder{filecount + 1}")
    os.makedirs(subfolder)
    regridder_path = os.path.join(subfolder, "regridder.nc")
    regridder.to_netcdf(regridder_path)

    fingerprint_path = os.path.join(subfolder, "fingerprint.json")
    with open(fingerprint_path, "w") as f:
        f.write(json.dumps(fingerprint))

def read_cache(cache_folder, fingerprint):
    os.makedirs(cache_folder, exist_ok=True)
    for subfolder in os.listdir(cache_folder):
        try:
            with open(os.path.join(cache_folder,subfolder,"fingerprint.json")) as f:
                cached_fingerprint = json.loads(f.read())
                if fingerprint == cached_fingerprint:
                    return os.path.join(cache_folder,subfolder,"regridder.nc")
        except:
            pass
    return None


def regrid(input_path, grid_path, output_path, method, limit=None, max_distance=None,
           output_distances_as=None, cache_folder=None):

    os.makedirs(output_path,exist_ok=True)
    if os.path.isdir(input_path):
        input_file_paths = list(map(lambda f: os.path.join(input_path,f),os.listdir(input_path)))
    else:
        input_file_paths = [input_path]

    idx = 1
    processed = 0
    total = len(input_file_paths)
    for input_file_path in input_file_paths:
        input_file_name = os.path.split(input_file_path)[1]
        output_file_path = os.path.join(output_path, input_file_name)
        if os.path.exists(output_file_path):
            print(f"skipping: {input_file_name} {idx}/{total} output already exists?")
            idx += 1
            continue
        else:
            print(f"processing: {input_file_name} {idx}/{total}")

        ds = xr.open_dataset(input_file_path)

        grid = xr.open_dataset(grid_path)

        lat_dims = set(ds.lat.dims)

        # track the names and original types of the variables to be regridded
        input_variables = []
        for name in ds.variables:
            if name not in {"lat","lon"}:
                v = ds.variables[name]
                if set(v.dims).issuperset(lat_dims):
                    input_variables.append((name,v.dtype))

        # construct a fingerprint which characterises the input and output grids
        # using the lat-lons for each corner (2D) or range end (1D)
        # this will be associated with a saved regridder, allowing regridder reuse
        fingerprint = {
            "input_shape": list(ds.lat.shape),
            "output_shape": list(grid.lat.shape)
        }

        fingerprint.update({
            "input_lat0": "%0.6f" % float(ds.lat[0, 0]),
            "input_lat1": "%0.6f" % float(ds.lat[-1, -1]),
            "input_lat2": "%0.6f" % float(ds.lat[-1, 0]),
            "input_lat3": "%0.6f" % float(ds.lat[0, -1]),
        })

        fingerprint.update({
            "input_lon0": "%0.6f" % float(ds.lon[0, 0]),
            "input_lon1": "%0.6f" % float(ds.lon[-1, -1]),
            "input_lon2": "%0.6f" % float(ds.lon[-1, 0]),
            "input_lon3": "%0.6f" % float(ds.lon[0, -1]),
        })

        if len(grid.lat.shape) == 2:
            fingerprint.update({
                "output_lat0": "%0.6f" % float(grid.lat[0, 0]),
                "output_lat1": "%0.6f" % float(grid.lat[-1, -1]),
                "output_lat2": "%0.6f" % float(grid.lat[-1, 0]),
                "output_lat3": "%0.6f" % float(grid.lat[0, -1])
            })

            fingerprint.update({
                "output_lon0": "%0.6f" % float(grid.lon[0, 0]),
                "output_lon1": "%0.6f" % float(grid.lon[-1, -1]),
                "output_lon2": "%0.6f" % float(grid.lon[-1, 0]),
                "output_lon3": "%0.6f" % float(grid.lon[0, -1])
            })
        else:
            fingerprint.update({
                "output_lat0": "%0.6f" % float(grid.lat[0]),
                "output_lat1": "%0.6f" % float(grid.lat[-1]),
            })

            fingerprint.update({
                "output_lon0": "%0.6f" % float(grid.lon[0]),
                "output_lon1": "%0.6f" % float(grid.lon[-1]),
            })


        cached_filename = None

        if cache_folder:
            cached_filename = read_cache(cache_folder,fingerprint)

        regridder = xe.Regridder(ds, grid, method, weights=cached_filename)
        ds_regridded = regridder(ds)

        # work out the distances between source and destination pixel centres, method=nearest_s2d only
        if method == "nearest_s2d" and (max_distance or output_distances_as):
            m_per_degree_lat = m_per_degree*np.cos(np.radians(grid.lat))
            distances = np.sqrt(np.power((grid.lon - ds_regridded.lon)*m_per_degree_lat,2)+np.power((grid.lat - ds_regridded.lat)*m_per_degree,2))
            if max_distance:
                # mask out output pixels too far from the nearest source pixels
                for (name,dtype) in input_variables:
                    ds_regridded[name] = ds_regridded[name].where(distances<max_distance,0 if np.issubdtype(dtype, np.integer) else np.nan)
            if output_distances_as:
                ds_regridded[output_distances_as] = xr.DataArray(distances, dims=ds_regridded.lat.dims)

        if method == "nearest_s2d":
            # nearest_s2d will write the lats/lons of the nearest source pixel, replace these with the target grid lat/lon
            ds_regridded["lat"] = xr.DataArray(grid.lat.data,dims=ds_regridded.lat.dims)
            ds_regridded["lon"] = xr.DataArray(grid.lon.data,dims=ds_regridded.lon.dims)

        copy_metadata_ds(ds_regridded,ds)

        encoding = {var_name: {"zlib": True, "complevel": 5} for var_name in list(ds_regridded.variables.keys()) if
                     var_name not in ["time"]}
        ds_regridded.to_netcdf(output_file_path, encoding=encoding)
        if cache_folder and cached_filename is None:
            write_cache(cache_folder,regridder,fingerprint)

        processed += 1
        if limit is not None and processed >= limit:
            print(f"stopping after processing {processed} scenes")
            break
        idx += 1

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path",
                        help="input data file to be regridded or folder containing files to be regridded")
    parser.add_argument("target_grid_path",
                        help="path to file defining grid lat/lon ontop which data is to be regridded")
    parser.add_argument("output_path", help="path to the output folder into which regridded files are be written")
    parser.add_argument("--method", default="nearest_s2d", help="regridding method passed to xesmf.Regridder")
    parser.add_argument("--limit", type=int, help="process only this many scenes (for testing)", default=None)
    parser.add_argument("--max-distance", type=float, help="specify the max distance (metres) for method=nearest_s2d",
                        default=100)
    parser.add_argument("--save-distances-as", type=str,
                        help="output distance to nearest source pixel in m to a variable with this name", default=None)
    parser.add_argument("--cache-folder", help="specify a folder to store and reuse regridders", default=None)

    args = parser.parse_args()

    regrid(input_path=args.input_path, grid_path=args.target_grid_path, output_path=args.output_path, method=args.method, limit=args.limit,
           max_distance=args.max_distance, output_distances_as=args.save_distances_as, cache_folder=args.cache_folder)


if __name__ == '__main__':
    main()
