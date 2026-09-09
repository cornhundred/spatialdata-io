"""Opt-in spatial tiling for a SpatialData store.

Two entry points, both a single call:

:func:`add_spatial_tiling`
    Add the profile to a store that already exists.
:func:`xenium_spatially_tiled`
    Read raw Xenium and write a tiled store in one go.

The one-shot path is internally ``read -> write -> tile``, because the tiling rewrites
*written* Parquet.

Everything here is additive and opt-in. A store that has been tiled is still an ordinary
SpatialData store: :func:`spatialdata.read_zarr` works unchanged, the canonical columns and
geometries are untouched, and a client that does not know about the profile simply ignores
the extra columns and the manifest.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from spatialdata_io.experimental.display_colors import add_cluster_colors, add_gene_colors
from spatialdata_io.experimental.expression_index import (
    add_gene_statistics,
    write_csc_layer,
)
from spatialdata_io.experimental.feature_catalog import FeatureCatalog
from spatialdata_io.experimental.manifest import (
    PROFILE_NAME,
    build_manifest,
    validate_manifest,
    write_manifest,
)
from spatialdata_io.experimental.points_parquet import (
    DisplayTransform,
    write_points_regular_grid,
)
from spatialdata_io.experimental.regular_grid import (
    DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    RegularGrid,
)
from spatialdata_io.experimental.shapes_parquet import (
    write_shapes_regular_grid,
)

__all__ = ["add_spatial_tiling", "xenium_spatially_tiled"]

#: Directory inside the store holding derived (non-canonical) profile assets.
PROFILE_DIR = "visualization"


#: Display colours cycled through when a channel has none assigned. First is blue, which
#: is the conventional nuclear stain colour and usually channel 0 (DAPI).
def _grid_for(points: Any, transform: DisplayTransform, tile_size_px: float) -> RegularGrid:
    """Derive the grid covering the points element in display pixel space."""
    x = points["x"].max().compute() if hasattr(points["x"].max(), "compute") else points["x"].max()
    y = points["y"].max().compute() if hasattr(points["y"].max(), "compute") else points["y"].max()
    px, py = transform.apply(np.array([float(x)]), np.array([float(y)]))
    return RegularGrid.from_bounds(0, 0, float(np.rint(px[0])), float(np.rint(py[0])), tile_size_px)


def add_spatial_tiling(
    store: str | Path,
    *,
    points_element: str = "transcripts",
    shapes_element: str | None = "cell_boundaries",
    table_element: str | None = "table",
    coordinate_system: str = "global",
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    feature_key: str = "feature_name",
    technology: str = "Xenium",
    index_expression: bool = True,
    cluster_column: str | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Add the regular-grid visualization profile to an existing SpatialData store.

    Re-runnable: running it again replaces the render columns and derived assets rather
    than duplicating them.

    Parameters
    ----------
    store
        Path to the ``.zarr`` store to tile, modified in place.
    points_element
        Name of the Points element holding transcripts.
    shapes_element
        Name of the Shapes element holding cell boundaries, or ``None`` to skip.
    table_element
        Name of the annotating table, used for the gene order and the CBG. ``None`` skips
        the CBG and derives feature codes from the observed features alone.
    coordinate_system
        Coordinate system defining display pixel space.
    tile_size_px
        Tile edge length in display pixels. The default of 250 gives roughly 20 cells per
        tile on Xenium-density tissue, which is the granularity the viewer fetches at.
    max_row_groups_per_file
        Row groups per chunk file.
    feature_key
        Column in the points element holding the feature name.
    technology
        Celldega technology string recorded in the manifest.
    index_expression
        Add per-gene statistics to ``var`` and a gene-major (CSC) copy of ``X`` as a layer,
        so a client can read one gene without downloading the whole matrix. Costs a second
        copy of the non-zeros.
    cluster_column
        Categorical ``obs`` column to colour cells by. Its palette is written to
        ``uns["<column>_colors"]``, the scanpy convention.
    compression
        Parquet compression codec.

    Returns
    -------
    The profile manifest.
    """
    import spatialdata

    store = Path(store)
    sdata = spatialdata.read_zarr(store)

    if points_element not in sdata.points:
        raise ValueError(f"points element {points_element!r} not found; have {list(sdata.points)}")
    points = sdata.points[points_element]
    table = sdata.tables[table_element] if table_element else None

    transform = DisplayTransform.from_element(points, coordinate_system)
    grid = _grid_for(points, transform, tile_size_px)

    if table is not None:
        catalog = FeatureCatalog.from_points_and_table(points, table, feature_key=feature_key)
    else:
        observed = points[feature_key]
        names = sorted(observed.cat.as_known().cat.categories) if hasattr(observed, "cat") else []
        catalog = FeatureCatalog(names=tuple(names), n_genes=len(names))

    profile_dir = store / PROFILE_DIR / PROFILE_NAME
    profile_dir.mkdir(parents=True, exist_ok=True)

    # The render columns go to a standalone file inside the profile directory. A viewer
    # reads every column of it, so it needs no column projection, and the canonical
    # element is left free of nested Arrow columns.
    transcripts = write_points_regular_grid(
        points,
        profile_dir / "trx",
        catalog=catalog,
        grid=grid,
        display_transform=transform,
        feature_key=feature_key,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        render_only=True,
        overwrite=True,
    )
    # The canonical element is re-ordered into tile row groups but keeps only its own
    # columns, so it still round-trips through SpatialData.write() and normal reads.
    write_points_regular_grid(
        points,
        store / "points" / points_element / "points.parquet",
        catalog=catalog,
        grid=grid,
        display_transform=transform,
        feature_key=feature_key,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        overwrite=True,
    )

    cell_segmentation = None
    if shapes_element:
        if shapes_element not in sdata.shapes:
            raise ValueError(f"shapes element {shapes_element!r} not found; have {list(sdata.shapes)}")
        shapes = sdata.shapes[shapes_element]
        shapes_transform = DisplayTransform.from_element(shapes, coordinate_system)
        cell_segmentation = write_shapes_regular_grid(
            shapes,
            profile_dir / "cell_seg",
            grid=grid,
            display_transform=shapes_transform,
            cell_index=list(table.obs_names) if table is not None else None,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            render_only=True,
            overwrite=True,
        )
        write_shapes_regular_grid(
            shapes,
            store / "shapes" / shapes_element / "shapes.parquet",
            grid=grid,
            display_transform=shapes_transform,
            cell_index=list(table.obs_names) if table is not None else None,
            max_row_groups_per_file=max_row_groups_per_file,
            compression=compression,
            overwrite=True,
        )

    # Gene-major access and a gene list both require reading every non-zero from a CSR
    # matrix. Precomputing the statistics and storing the transpose turns "download the
    # whole matrix" into "read two chunks", which is what makes this scale.
    expression_index: dict[str, Any] | None = None
    if index_expression and table is not None and table_element:
        import scipy.sparse as sp

        stats = add_gene_statistics(table)
        colors = add_gene_colors(table)
        cluster_colors = add_cluster_colors(table, cluster_column) if cluster_column else None
        # SpatialData refuses to overwrite an element inside the store it was read from
        # (scverse/spatialdata#520), so the table is deleted and rewritten. The statistics
        # pass has already materialised X, so nothing is read back from the deleted path.
        csc = table.X.tocsc() if sp.issparse(table.X) else None
        sdata.delete_element_from_disk(table_element)
        sdata.write_element(table_element)

        # Written after the table, and directly, because the chunking is the whole point:
        # AnnData sizes chunks for whole-matrix reads, which costs 24x too much per gene.
        layer = write_csc_layer(store / "tables" / table_element, csc) if csc is not None else None
        expression_index = {
            "var_statistics": stats,
            **({"csc": layer} if layer else {}),
            **({"gene_colors": colors} if colors else {}),
            **({"cluster_colors": cluster_colors} if cluster_colors else {}),
        }

    manifest = build_manifest(
        grid=grid,
        technology=technology,
        transcripts=transcripts,
        cell_segmentation=cell_segmentation,
        # Gene and cell metadata, expression and images are all read from the store itself,
        # so the profile declares where the store is rather than duplicating its contents.
        feature_catalog={
            "n_genes": catalog.n_genes,
            "extra_features": list(catalog.names[catalog.n_genes :]),
        },
        spatialdata={
            "store_url": "../..",
            "table": table_element or "table",
            "native": ["metadata", "cbg", "images"],
            **({"cluster_column": cluster_column} if cluster_column else {}),
            **({"expression_index": expression_index} if expression_index else {}),
        },
        source={
            "store": store.name,
            "points_element": points_element,
            "shapes_element": shapes_element,
            "table_element": table_element,
            "coordinate_system": coordinate_system,
            "tile_size_px": tile_size_px,
        },
    )
    validate_manifest(manifest, base_path=profile_dir)
    write_manifest(manifest, profile_dir)
    return manifest


def xenium_spatially_tiled(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    tile_size_px: float = 250.0,
    max_row_groups_per_file: int = DEFAULT_MAX_ROW_GROUPS_PER_FILE,
    compression: str = "zstd",
    overwrite: bool = False,
    tiling: dict[str, Any] | None = None,
    **xenium_kwargs: Any,
) -> dict[str, Any]:
    """Read raw Xenium data and write a spatially tiled SpatialData store in one call.

    Parameters
    ----------
    raw_path
        Directory of raw Xenium output.
    output_path
        Destination ``.zarr`` store.
    tile_size_px
        Tile edge length in display pixels.
    max_row_groups_per_file
        Row groups per chunk file.
    compression
        Parquet compression codec.
    overwrite
        Replace ``output_path`` if it exists.
    tiling
        Extra keyword arguments for :func:`add_spatial_tiling`, for example
        ``{"shapes_element": "cell_polygons"}``. Kept separate from ``xenium_kwargs``
        because the two functions have distinct option sets.
    xenium_kwargs
        Forwarded to :func:`spatialdata_io.xenium`.

    Returns
    -------
    The profile manifest.
    """
    from spatialdata_io.readers.xenium import xenium

    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
        shutil.rmtree(output_path)

    sdata = xenium(raw_path, **xenium_kwargs)
    sdata.write(output_path)
    return add_spatial_tiling(
        output_path,
        tile_size_px=tile_size_px,
        max_row_groups_per_file=max_row_groups_per_file,
        compression=compression,
        **(tiling or {}),
    )
