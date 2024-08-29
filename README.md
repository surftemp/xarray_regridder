# xarray_regridder

A flexible regridding utility API nd CLI based on xarray and pyproj.  Maps from input data grid(s) to a regular target grid.  

Note - the target grid must have equally spaced, one-dimensional coordinates.

## Environment

```
conda create -n xarray_regridder_env python=3.10
conda activate xarray_regridder_env
conda install netcdf4 xarray pyproj
pip install git+https://github.com/surftemp/netcdf2html.git
```

## supported regrdding modes

this tool will analyse the set of source pixels that map to each destination pixel in the regular target grid.

| mode name | function                                                                      |
|-----------|-------------------------------------------------------------------------------|
 | min       | retain the min value of all source pixels that map to a destination pixel     |
| max       | retain the max value of all source pixels that map to a destination pixel     |
| mean      | retain the min value of all source pixels that map to a destination pixel     |
| nearest   | retain the value of the nearest source pixels that map to a destination pixel |

## input data

input data can be specified as an input netcdf4 file or a folder containing multiple netcdf4 files

## Examples

### Convert from data imported from landsat (see see https://github.com/surftemp/landsat_importer) to a regular lat-lon grid (target_grid.nc)

Use the nearest neighbour mode to map variables B2, B3, B4 to the target grid:

```
xarray_regridder landsat_imported.nc target_grid.nc regridded_output.nc --source-crs 4326 --source-x lon --source-y lat --target-crs 4326 --target-x lon --target-y lat --variables B2:nearest:B2 B3:nearest:B3 B4:nearest:B4
```

Here, target_grid.nc has 1-dimensional lat and lon variables:

```
ncdump -h target_grid.nc 
netcdf grid {
dimensions:
	nj = 2000 ;
	ni = 2000 ;
variables:
	double lat(nj) ;
		lat:_FillValue = NaN ;
		lat:units = "degrees_north" ;
		lat:standard_name = "latitude" ;
	double lon(ni) ;
		lon:_FillValue = NaN ;
		lon:units = "degrees_east" ;
		lon:standard_name = "longitude" ;
}
```