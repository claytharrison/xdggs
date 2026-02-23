import numpy as np
import pytest
import xarray as xr
from xarray.core.indexes import PandasIndex

from xdggs import fibgrid

# A small selection of known Fibonacci grid points at res=12.5 km
# (GPI 0 is the first point at lon=0, lat=0)
cell_ids = [
    np.array([0]),
    np.array([0, 1226431]),
    np.array([0, 1226431, 1162085]),
]

# Expected (lon, lat) centres for the above cell ids at res=12.5
cell_centers = [
    np.array([[0.0, 0.0]]),
    np.array([[0.0, 0.0], [15.428348, 48.203693]]),
    np.array([[0.0, 0.0], [15.428348, 48.203693], [10.013912, 44.96483]]),
]

dims = ["cells", "zones"]
resolutions = [6.25, 12.5, 25.0]
levels = [2, 1, 0]


class TestFibGridInfo:
    @pytest.mark.parametrize(
        ["resolution", "error"],
        (
            (12.5, None),
            (25.0, None),
            (6.25, None),
            (1.0, ValueError("resolution must be one of")),
            (0.0, ValueError("resolution must be one of")),
            (-1.0, ValueError("resolution must be one of")),
            (1e9, ValueError("resolution must be one of")),
        ),
    )
    def test_init(self, resolution, error):
        if error is not None:
            with pytest.raises(type(error), match=str(error)):
                fibgrid.FibGridInfo(resolution=resolution)
            return

        actual = fibgrid.FibGridInfo(resolution=resolution)
        assert actual.resolution == resolution

    @pytest.mark.parametrize(
        ["resolution", "expected_level"],
        list(zip(resolutions, levels)),
    )
    def test_level_from_resolution(self, resolution, expected_level):
        info = fibgrid.FibGridInfo(resolution=resolution)
        assert info.level == expected_level

    @pytest.mark.parametrize(
        ["mapping", "expected_resolution"],
        (
            ({"resolution": 12.5}, 12.5),
            ({"resolution": 25.0}, 25.0),
            ({"level": 0}, 25.0),
            ({"level": 1}, 12.5),
            ({"level": 2}, 6.25),
            ({"resolution": 99.0}, ValueError("resolution must be one of")),
        ),
    )
    def test_from_dict(self, mapping, expected_resolution):
        if isinstance(expected_resolution, Exception):
            with pytest.raises(
                type(expected_resolution), match=str(expected_resolution)
            ):
                fibgrid.FibGridInfo.from_dict(mapping)
            return

        actual = fibgrid.FibGridInfo.from_dict(mapping)
        assert actual.resolution == expected_resolution

    def test_roundtrip(self):
        mapping = {"grid_name": "fibgrid", "resolution": 12.5}
        grid = fibgrid.FibGridInfo.from_dict(mapping)
        actual = grid.to_dict()
        assert actual == mapping

    @pytest.mark.parametrize(
        ["cell_ids_arr", "expected_centers"],
        list(zip(cell_ids, cell_centers)),
    )
    def test_cell_ids2geographic(self, cell_ids_arr, expected_centers):
        grid = fibgrid.FibGridInfo(resolution=12.5)
        actual = grid.cell_ids2geographic(cell_ids_arr)

        assert isinstance(actual, tuple) and len(actual) == 2
        # lon and lat returned separately; expected_centers is (N, 2) with lon first
        np.testing.assert_allclose(actual[0], expected_centers[:, 0], atol=1e-4)
        np.testing.assert_allclose(actual[1], expected_centers[:, 1], atol=1e-4)

    @pytest.mark.parametrize(
        ["expected_ids", "centers"],
        list(zip(cell_ids, cell_centers)),
    )
    def test_geographic2cell_ids(self, expected_ids, centers):
        grid = fibgrid.FibGridInfo(resolution=12.5)
        actual = grid.geographic2cell_ids(
            lon=centers[:, 0], lat=centers[:, 1]
        )
        np.testing.assert_equal(actual, expected_ids)

    def test_cell_boundaries_raises(self):
        grid = fibgrid.FibGridInfo(resolution=12.5)
        with pytest.raises(NotImplementedError):
            grid.cell_boundaries(np.array([0]))


class TestFibGridIndex:
    @pytest.mark.parametrize("resolution", resolutions)
    def test_from_variables(self, resolution):
        var = xr.Variable(
            "cells",
            np.array([0, 1000]),
            {"grid_name": "fibgrid", "resolution": resolution},
        )
        index = fibgrid.FibGridIndex.from_variables({"cell_ids": var}, options={})
        assert isinstance(index, fibgrid.FibGridIndex)
        assert index.grid_info.resolution == resolution

    def test_from_variables_with_options(self):
        """options dict should override variable attributes."""
        var = xr.Variable(
            "cells",
            np.array([0, 1000]),
            {"grid_name": "fibgrid", "resolution": 12.5},
        )
        index = fibgrid.FibGridIndex.from_variables(
            {"cell_ids": var}, options={"resolution": 25.0}
        )
        assert index.grid_info.resolution == 25.0

    def test_repr_inline(self):
        grid_info = fibgrid.FibGridInfo(resolution=12.5)
        index = fibgrid.FibGridIndex(np.array([0, 1000]), "cells", grid_info)
        assert "12.5" in index._repr_inline_(100)

    def test_invalid_grid_info_type(self):
        from xdggs.healpix import HealpixInfo

        with pytest.raises(ValueError, match="invalid type"):
            fibgrid.FibGridIndex(
                np.array([0]),
                "cells",
                HealpixInfo(level=2),
            )

    def test_cell_boundaries_raises_via_index(self):
        grid_info = fibgrid.FibGridInfo(resolution=12.5)
        index = fibgrid.FibGridIndex(np.array([0, 1000]), "cells", grid_info)
        with pytest.raises(NotImplementedError):
            index.cell_boundaries()


class TestFibGridXarrayIntegration:
    """End-to-end tests using the xarray accessor."""

    def _make_dataset(self, resolution=12.5):
        gpis = np.array([0, 1226431, 1162085])
        return xr.Dataset(
            {"data": ("cell_ids", np.arange(len(gpis), dtype=float))},
            coords={
                "cell_ids": (
                    "cell_ids",
                    gpis,
                    {"grid_name": "fibgrid", "resolution": resolution},
                )
            },
        )

    def test_decode(self):
        ds = self._make_dataset()
        decoded = ds.dggs.decode()
        assert isinstance(decoded.xindexes["cell_ids"], fibgrid.FibGridIndex)

    def test_assign_latlon_coords(self):
        ds = self._make_dataset()
        decoded = ds.dggs.decode()
        with_coords = decoded.dggs.assign_latlon_coords()
        assert "latitude" in with_coords.coords
        assert "longitude" in with_coords.coords
        np.testing.assert_allclose(
            with_coords["latitude"].values, [0.0, 48.203693, 44.96483], atol=1e-4
        )

    def test_sel_latlon(self):
        ds = self._make_dataset()
        decoded = ds.dggs.decode()
        result = decoded.dggs.sel_latlon(latitude=48.2, longitude=15.5)
        # Should select cell 1226431 (nearest to lon=15.5, lat=48.2)
        assert int(result["cell_ids"].values) == 1226431

    def test_grid_info_accessible(self):
        ds = self._make_dataset()
        decoded = ds.dggs.decode()
        info = decoded.dggs.grid_info
        assert isinstance(info, fibgrid.FibGridInfo)
        assert info.resolution == 12.5
