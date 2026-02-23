from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Self

import numpy as np
import xarray as xr
from xarray.indexes import PandasIndex

from xdggs.ellipsoid import Ellipsoid, Sphere
from xdggs.grid import DGGSInfo, translate_parameters
from xdggs.index import DGGSIndex
from xdggs.utils import _extract_cell_id_variable, register_dggs

# Valid resolutions in kilometres supported by the fibgrid package
_VALID_RESOLUTIONS: list[float] = [6.25, 12.5, 25.0]

# Map resolution → level (coarsest=0, finest=2)
_RESOLUTION_TO_LEVEL: dict[float, int] = {25.0: 0, 12.5: 1, 6.25: 2}
_LEVEL_TO_RESOLUTION: dict[int, float] = {v: k for k, v in _RESOLUTION_TO_LEVEL.items()}


@lru_cache(maxsize=len(_VALID_RESOLUTIONS))
def _get_fibgrid(resolution: float):
    """Return a cached FibGrid instance for the given resolution."""
    from fibgrid.realization import FibGrid

    return FibGrid(resolution)


@dataclass(frozen=True)
class FibGridInfo(DGGSInfo):
    """Grid information container for the TU Wien Fibonacci grid.

    The Fibonacci grid places points quasi-uniformly on the sphere using the
    golden-angle spiral.  Unlike most DGGS implementations, cells have no
    explicit boundaries; the natural way to query this grid is by finding the
    nearest grid point to a given (lon, lat) coordinate.

    Parameters
    ----------
    resolution : float
        Grid resolution in kilometres.  Must be one of ``6.25``, ``12.5``, or
        ``25.0``.

    Notes
    -----
    The ``level`` attribute is derived automatically from ``resolution`` and
    follows the convention used by the rest of xdggs (higher level = finer
    grid):

    - ``resolution=25.0``  → ``level=0``
    - ``resolution=12.5``  → ``level=1``
    - ``resolution=6.25``  → ``level=2``
    """

    # Override parent's ``level``: set in ``__post_init__`` from ``resolution``.
    level: int = field(default=0, init=False, repr=False, compare=False)

    resolution: float = field(kw_only=True)
    """float : Grid resolution in kilometres (6.25, 12.5 or 25.0)."""

    # Override parent's ``ellipsoid`` to keep default; Fibonacci grid is WGS84.
    ellipsoid: str | Sphere | Ellipsoid = field(default="sphere", kw_only=True)

    valid_parameters: ClassVar[dict[str, Any]] = {
        "resolution": _VALID_RESOLUTIONS,
    }

    def __post_init__(self) -> None:
        if self.resolution not in _VALID_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_VALID_RESOLUTIONS}, got {self.resolution}"
            )
        object.__setattr__(self, "level", _RESOLUTION_TO_LEVEL[self.resolution])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        """Construct a :class:`FibGridInfo` from a mapping of attributes.

        Parameters
        ----------
        mapping : dict of str to any
            Grid attributes.  Recognised keys are ``resolution`` (km) and
            ``level``.

        Returns
        -------
        grid_info : FibGridInfo
        """
        translations = {
            "resolution": ("resolution", float),
            "level": ("resolution", lambda lvl: _LEVEL_TO_RESOLUTION[int(lvl)]),
        }
        params = translate_parameters(mapping, translations)
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        """Dump the normalised grid parameters.

        Returns
        -------
        mapping : dict of str to any
        """
        return {"grid_name": "fibgrid", "resolution": self.resolution}

    # ------------------------------------------------------------------
    # Grid operations
    # ------------------------------------------------------------------

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert grid-point indices (GPIs) to geographic coordinates.

        Parameters
        ----------
        cell_ids : array-like of int
            Grid-point indices.

        Returns
        -------
        lon : numpy.ndarray
            Longitude values in degrees.
        lat : numpy.ndarray
            Latitude values in degrees.
        """
        grid = _get_fibgrid(self.resolution)
        lons, lats = grid.gpi2lonlat(np.asarray(cell_ids))
        return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)

    def geographic2cell_ids(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> np.ndarray:
        """Find the nearest Fibonacci grid point for each (lon, lat) pair.

        Parameters
        ----------
        lon : array-like of float
            Longitude values in degrees.
        lat : array-like of float
            Latitude values in degrees.

        Returns
        -------
        cell_ids : numpy.ndarray of int
            Grid-point indices of the nearest grid point for each input
            coordinate.
        """
        grid = _get_fibgrid(self.resolution)
        gpis, _ = grid.find_nearest_gpi(
            np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        )
        return np.asarray(gpis)

    def cell_boundaries(self, cell_ids: Any, backend: str = "shapely") -> None:
        """Not implemented – the Fibonacci grid has no cell-boundary data.

        Parameters
        ----------
        cell_ids : array-like
            Unused.  Present for interface compatibility with :class:`DGGSInfo`.
        backend : str
            Unused.  Present for interface compatibility with :class:`DGGSInfo`.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "The Fibonacci grid does not define cell boundaries."
        )

    def zoom_to(self, cell_ids: Any, level: int) -> None:
        """Not implemented – the Fibonacci grid has no hierarchical structure.

        Parameters
        ----------
        cell_ids : array-like
            Unused.  Present for interface compatibility with :class:`DGGSInfo`.
        level : int
            Unused.  Present for interface compatibility with :class:`DGGSInfo`.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "The Fibonacci grid does not support zoom_to (no hierarchical structure)."
        )


@register_dggs("fibgrid")
class FibGridIndex(DGGSIndex):
    """Xarray index for the TU Wien Fibonacci grid.

    See Also
    --------
    FibGridInfo : grid parameters container
    """

    def __init__(
        self,
        cell_ids: Any | PandasIndex,
        dim: str,
        grid_info: FibGridInfo,
    ) -> None:
        if not isinstance(grid_info, FibGridInfo):
            raise ValueError(
                f"grid info object has an invalid type: {type(grid_info)}"
            )
        super().__init__(cell_ids, dim, grid_info)

    @classmethod
    def from_variables(
        cls: type["FibGridIndex"],
        variables: Mapping[Any, xr.Variable],
        *,
        options: Mapping[str, Any],
    ) -> "FibGridIndex":
        _, var, dim = _extract_cell_id_variable(variables)
        grid_info = FibGridInfo.from_dict(var.attrs | options)
        return cls(var.data, dim, grid_info)

    def _replace(self, new_pd_index: PandasIndex) -> "FibGridIndex":
        return type(self)(new_pd_index, self._dim, self._grid)

    @property
    def grid_info(self) -> FibGridInfo:
        return self._grid

    def _repr_inline_(self, max_width: int) -> str:
        return f"FibGridIndex(resolution={self._grid.resolution}km)"

