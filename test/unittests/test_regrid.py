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

import unittest
import xarray as xr
import numpy as np
import math

import logging

from xarray_regridder.api.regridder import Regridder as Regrid

logging.basicConfig(level=logging.INFO)

class TestCase(unittest.TestCase):

    def test_min_max_mean(self):
        source_step = 10
        source_x_min = 1005
        source_x_max = 2005
        source_y_min = 1005
        source_y_max = 2005

        dest_step = 20
        dest_x_min = 1010
        dest_x_max = 2010
        dest_y_min = 1010
        dest_y_max = 2010

        source_ds = xr.Dataset()
        source_y = np.arange(source_y_min, source_y_max, source_step)
        source_x = np.arange(source_x_min, source_x_max, source_step)

        source_h = source_y.shape[0]
        source_w = source_x.shape[0]

        source_ds["y"] = xr.DataArray(source_y, dims=("y",), attrs={"units": "m"})
        source_ds["x"] = xr.DataArray(source_x, dims=("x",), attrs={"units": "m"})
        source_ds["data"] = xr.DataArray(np.random.rand(source_h,source_w),dims = ("y","x"))

        grid_ds = xr.Dataset()
        grid_y = np.arange(dest_y_min, dest_y_max, dest_step)
        grid_x = np.arange(dest_x_min, dest_x_max, dest_step)

        grid_ds["y"] = xr.DataArray(grid_y, dims=("y",), attrs={"units": "m"})
        grid_ds["x"] = xr.DataArray(grid_x, dims=("x",), attrs={"units": "m"})

        grid_h = grid_y.shape[0]
        grid_w = grid_x.shape[0]

        self.assertEqual(2*grid_h, source_h)
        self.assertEqual(2*grid_w, source_w)

        r = Regrid(grid_ds=grid_ds, variables=["data:min","data:max","data:mean"],
                 source_x="x", source_y="y", source_crs=27700, target_x="x", target_y="y", target_crs=27700)

        r.ingest(source_ds)

        output_ds, output_encodings = r.get_output()

        expected_mins = source_ds["data"].coarsen(dim={"x":2,"y":2}).min()
        expected_maxes = source_ds["data"].coarsen(dim={"x": 2, "y": 2}).max()
        expected_means = source_ds["data"].coarsen(dim={"x": 2, "y": 2}).mean()

        import numpy.testing as npt
        npt.assert_allclose(output_ds["data_min"].data,expected_mins)
        npt.assert_allclose(output_ds["data_max"].data,expected_maxes)
        npt.assert_allclose(output_ds["data_mean"].data, expected_means)

    def test_nearest(self):
        source_step = 10
        source_x_min = 100
        source_x_max = 200
        source_y_min = 100
        source_y_max = 200

        dest_step = 20
        dest_x_min = 101
        dest_x_max = 201
        dest_y_min = 101
        dest_y_max = 201

        source_ds = xr.Dataset()
        source_y = np.arange(source_y_min, source_y_max, source_step)
        source_x = np.arange(source_x_min, source_x_max, source_step)

        source_h = source_y.shape[0]
        source_w = source_x.shape[0]

        source_ds["y"] = xr.DataArray(source_y, dims=("y",), attrs={"units": "m"})
        source_ds["x"] = xr.DataArray(source_x, dims=("x",), attrs={"units": "m"})
        source_ds["data"] = xr.DataArray((100*np.random.rand(source_h,source_w)).astype(int),dims = ("y","x"))

        print("source dimension: "+str((source_h,source_w)))
        print(source_ds)

        grid_ds = xr.Dataset()
        grid_y = np.arange(dest_y_min, dest_y_max, dest_step)
        grid_x = np.arange(dest_x_min, dest_x_max, dest_step)

        grid_ds["y"] = xr.DataArray(grid_y, dims=("y",), attrs={"units": "m"})
        grid_ds["x"] = xr.DataArray(grid_x, dims=("x",), attrs={"units": "m"})

        grid_h = grid_y.shape[0]
        grid_w = grid_x.shape[0]

        print("target dimension: " + str((grid_h, grid_w)))

        r = Regrid(grid_ds=grid_ds, variables=["data:nearest","data:distance:nearest_distance"],
                 source_x="x", source_y="y", source_crs=27700, target_x="x", target_y="y", target_crs=27700)

        r.ingest(source_ds, stride=4)
        output_ds, output_encodings = r.get_output()

        expected_nearest = source_ds["data"].isel(x=slice(None,None,2),y=slice(None,None,2))

        expected_distance = math.sqrt(2)
        print(output_ds["nearest_distance"])

        import numpy.testing as npt
        npt.assert_allclose(output_ds["data_nearest"].data,expected_nearest)
        npt.assert_allclose(output_ds["nearest_distance"].data, expected_distance)







if __name__ == '__main__':

    unittest.main()