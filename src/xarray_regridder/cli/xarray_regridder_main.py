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
import os
import logging
import time

from xarray_regridder.api.regridder import Regridder

RETRY_DELAY = 60

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path",
                        help="input data file to be regridded or folder containing .nc files to be regridded")
    parser.add_argument("target_grid_path",
                        help="path to file defining grid with 1D lat/lon onto which data is to be regridded")
    parser.add_argument("output_path",
                        help="path to file to hold combined regridded data OR folder to hold individually regridded files")

    parser.add_argument("--target-y", type=str, help="target grid y coordinate variable nane", default="lat")
    parser.add_argument("--target-x", type=str, help="target grid x coordinate variable nane", default="lon")
    parser.add_argument("--target-crs", type=int, help="target CRS, given as an EPSG number", default=4326)

    parser.add_argument("--source-y", type=str, help="input y coordinate variable nane", default="lat")
    parser.add_argument("--source-x", type=str, help="input x coordinate variable nane", default="lon")
    parser.add_argument("--source-crs", type=int, help="source CRS, given as an EPSG number", default=4326)

    parser.add_argument("--variables", nargs="+", help="Specify variables to process, for nearest use NAME, for other modes (min,max,mean) use NAME:MODE, to specify the output variable name use NAME:MODE:OUTPUT_NAME")

    parser.add_argument("--limit", type=int, help="process only this many scenes (for testing)", default=None)
    parser.add_argument("--stride", type=int, help="set the stride", default=1)
    parser.add_argument("--nr-retries", type=int, help="use this many re-attempts to process each input file", default=0)
    parser.add_argument("--attr", nargs=2, action="append", type=str, metavar=("NAME","VALUE"), help="add global attributes", required=False)
    parser.add_argument("--chunk-sizes", nargs=2, type=str, metavar=("Y-DIMENSION-SIZE", "X-DIMENSION-SIZE"),
                        help="set the chunk sizes for regridded variables", required=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("regridder")

    grid_ds = xr.open_dataset(args.target_grid_path)

    regridder = Regridder(
        variables=args.variables,
        grid_ds=grid_ds,
        source_x=args.source_x,
        source_y=args.source_y,
        source_crs=args.source_crs,
        target_x=args.target_x,
        target_y=args.target_y,
        target_crs=args.target_crs)

    output_individual_files = False
    if os.path.isdir(args.output_path):
        if os.path.isdir(args.input_path):
            output_individual_files = True
        else:
            raise Exception("If the output path is a folder, the input path should also be")
    else:
        output_folder = os.path.split(args.output_path)[0]
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

    if os.path.isdir(args.input_path):

        input_file_paths = list(map(lambda f: os.path.join(args.input_path, f),
                                    filter(lambda name: name.endswith(".nc"),os.listdir(args.input_path))))
    else:
        output_individual_files = True
        input_file_paths = [args.input_path]

    idx = 1
    processed = 0
    total = len(input_file_paths)

    regridder.reset()

    for input_file_path in input_file_paths:
        ingested = False
        time_da = None
        for retry in range(0, args.nr_retries + 1):
            try:
                input_file_name = os.path.split(input_file_path)[1] if input_file_path else "<input dataset>"

                logger.info(f"processing: {input_file_name} {idx}/{total}")

                ds = xr.open_dataset(input_file_path)

                if "time" in ds:
                    time_da = ds["time"]

                ingested = regridder.ingest(ds, args.stride)

            except Exception as ex:
                logger.exception(f"Error processing: {input_file_name} : {str(ex)}")
                time.sleep(RETRY_DELAY)

        if not ingested:
            logger.error(f"Unable to process: {input_file_name}")
        else:
            if output_individual_files:
                output_ds, encodings = regridder.get_output(time_da=time_da, chunk_sizes=args.chunk_sizes)
                if args.attr:
                    for (name,value) in args.attr:
                        output_ds.attrs[name] = value
                if os.path.isdir(args.output_path):
                    output_path = os.path.join(args.output_path, input_file_name)
                else:
                    output_path = args.output_path

                for retry in range(0, args.nr_retries + 1):
                    try:
                        logger.info(f"writing: {output_path}")
                        output_ds.to_netcdf(output_path, encoding=encodings)
                        break
                    except Exception as ex:
                        logger.error(f"Error writing: {output_path} : {str(ex)}")
                        time.sleep(RETRY_DELAY)
                regridder.reset()

            processed += 1
            if args.limit is not None and processed >= args.limit:
                logger.info(f"stopping after processing {processed} input files")
                break
            idx += 1

    if not output_individual_files:
        # output accumulated results from merging all input files
        output_ds, encodings = regridder.get_output(chunk_sizes=args.chunk_sizes)

        if args.attr:
            for (name, value) in args.attr:
                output_ds.attrs[name] = value

        for retry in range(0, args.nr_retries + 1):
            try:
                logger.info(f"writing: {args.output_path}")
                output_ds.to_netcdf(args.output_path, encoding=encodings)
                break
            except Exception as ex:
                logger.error(f"Error writing: {args.output_path} : {str(ex)}")
                time.sleep(RETRY_DELAY)


if __name__ == '__main__':
    main()
