# Using xdggs with the Fibonacci Grid

This guide shows how to use `xdggs` with data on the
[TU Wien Fibonacci grid](https://github.com/TUW-GEO/fibgrid), a quasi-uniform
point distribution useful for satellite-derived soil-moisture products and
other global remote-sensing datasets.

## Background

The Fibonacci grid places ~N points quasi-uniformly on the sphere using the
golden-angle spiral.  Three resolutions are available:

| Resolution (km) | Level | Approx. number of points |
|-----------------|-------|--------------------------|
| 6.25            | 2     | ~6 600 000               |
| 12.5            | 1     | ~3 300 000               |
| 25.0            | 0     | ~860 000                 |

Unlike hexagonal or triangular grids, the Fibonacci grid is a **point-cloud**:
cells have no explicit boundaries and there is no connectivity between
neighbours.  Spatial queries therefore use **nearest-neighbour** lookup rather
than point-in-polygon containment.

## Installation

The Fibonacci grid support requires the `fibgrid` package:

```bash
pip install xdggs fibgrid
```

## Typical dataset layout

Many datasets (e.g. ASCAT, H-SAF) ship data as NetCDF/Zarr files with

```
Dimensions:  (gpi: 3 300 001, time: T)
Coordinates:
  * gpi      (gpi)  int64  cell_ids
  * time     (time) datetime64
```

The important point is that the **dimension is called `gpi`** but the
**coordinate is called `cell_ids`**.  `xdggs` works with the coordinate name,
so you need to ensure the cell-id coordinate has that name (or tell `decode`
which coordinate to use via the `name` argument).

## Step-by-step guide

### 1. Load your data

```python
import numpy as np
import xarray as xr
import xdggs  # registers the FibGridIndex

# Simulate a (gpi, time) dataset
n_gpi = 5
gpis = np.array([0, 1226431, 1162085, 430000, 860000])   # Fibonacci grid point indices
times = np.arange("2020-01", "2020-04", dtype="datetime64[M]")

ds = xr.Dataset(
    {"smi": (["gpi", "time"], np.random.rand(n_gpi, len(times)))},
    coords={
        "cell_ids": ("gpi", gpis, {        # <── coordinate name must be "cell_ids"
            "grid_name": "fibgrid",
            "resolution": 12.5,            # km
        }),
        "time": times,
    },
)
print(ds)
```

```
<xarray.Dataset>
Dimensions:   (gpi: 5, time: 3)
Coordinates:
    cell_ids  (gpi) int64 ...
  * time      (time) datetime64[M] 2020-01 2020-02 2020-03
Dimensions without coordinates: gpi
Data variables:
    smi       (gpi, time) float64 ...
```

### 2. Decode the DGGS index

Calling `.dggs.decode()` attaches an xdggs index to the `cell_ids` coordinate
so that spatial operations become available.

```python
ds = ds.dggs.decode()
print(ds)
```

```
<xarray.Dataset>
Dimensions:   (gpi: 5, time: 3)
Coordinates:
  * cell_ids  (gpi) int64 ...
  * time      (time) datetime64[M] ...
Indexes:
    cell_ids  FibGridIndex(resolution=12.5km)
Data variables:
    smi       (gpi, time) float64 ...
```

The `cell_ids` coordinate is now indexed by a `FibGridIndex`.

> **Tip** – if your dataset stores the GPIs in a coordinate with a *different*
> name (e.g. `"gpi"` or `"location"`), pass `name="your_coord"`:
>
> ```python
> ds = ds.dggs.decode(name="gpi")
> ```

### 3. Add latitude / longitude coordinates

```python
ds = ds.dggs.assign_latlon_coords()
print(ds.coords)
```

```
Coordinates:
  * cell_ids   (gpi) int64 ...
    latitude   (gpi) float64 ...
    longitude  (gpi) float64 ...
  * time       (time) datetime64[M] ...
```

### 4. Select grid points from lat/lon

`sel_latlon` finds the nearest Fibonacci grid point for every supplied
(latitude, longitude) pair and returns the matching data:

```python
# Single point
vienna = ds.dggs.sel_latlon(latitude=48.2, longitude=16.4)
print(vienna)

# Multiple points
lats = np.array([48.2, 51.5, 40.7])
lons = np.array([16.4,  -0.1, -74.0])
cities = ds.dggs.sel_latlon(latitude=lats, longitude=lons)
print(cities)
```

### 5. Work with the time dimension normally

Because `xdggs` only indexes the `gpi` dimension, the `time` dimension is
untouched.  All standard xarray operations work as expected:

```python
# Time-mean per grid point
mean_smi = ds["smi"].mean("time")

# Select a single time step, then query spatially
ds.sel(time="2020-01").dggs.sel_latlon(latitude=48.2, longitude=16.4)

# Group by month
monthly = ds["smi"].groupby("time.month").mean()
```

### 6. Query grid info

```python
print(ds.dggs.grid_info)
# FibGridInfo(ellipsoid='sphere', resolution=12.5)

print(ds.dggs.grid_info.resolution)  # 12.5 (km)
print(ds.dggs.grid_info.level)       # 1
```

## Decoding from dataset attributes

If your dataset already has the required attributes on the `cell_ids`
coordinate (e.g. written by `xdggs.encode`), you can decode it with:

```python
ds = xr.open_dataset("your_file.nc")
ds = ds.dggs.decode()
```

The minimum required attributes on the coordinate are:

```python
{
    "grid_name": "fibgrid",
    "resolution": 12.5,   # or "level": 1
}
```

You can also pass the grid info explicitly to override or supplement what is
in the file:

```python
import xdggs

ds = ds.dggs.decode(grid_info=xdggs.FibGridInfo(resolution=12.5))
# or equivalently
ds = ds.dggs.decode(grid_info={"grid_name": "fibgrid", "resolution": 12.5})
```

## Encoding (writing) back to file

```python
ds_encoded = xdggs.encode(ds)          # writes grid_name/resolution as attrs
ds_encoded.to_netcdf("output.nc")
```

## Notes on the nearest-neighbour paradigm

The Fibonacci grid is a **point-cloud** (a set of discrete locations on the
sphere) rather than a partition of the sphere into non-overlapping cells.  As
a result:

- There is no hierarchical structure, so `ds.dggs.zoom_to(level)` raises
  `NotImplementedError`.
- There are no polygon boundaries for cells, so `ds.dggs.cell_boundaries()`
  raises `NotImplementedError`.

`sel_latlon` handles both of these gracefully by using nearest-neighbour
lookup instead of point-in-polygon containment.  Everything else –
`assign_latlon_coords`, `cell_centers`, `decode`, `encode` – works as with
other grids.
