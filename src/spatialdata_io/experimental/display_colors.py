"""Assign display colours to genes and to categorical cell annotations.

There is no biologically correct colour for a gene, so these are invented either way. The
question is only where they live, and a ``var`` column is the better answer than a
viewer-specific file: it round-trips through AnnData, scanpy and any other tool can see it,
and it survives a re-read.

Worth being explicit that a gene colour column is a **new convention**. AnnData has
``uns["<column>_colors"]`` for *obs* categoricals -- which :func:`add_cluster_colors`
follows exactly -- but nothing for genes.

The palette matches the fallback Celldega generates when a store has no colours, so a store
looks the same whether or not this ran.
"""

from __future__ import annotations

import colorsys
from typing import Any

__all__ = ["add_gene_colors", "add_cluster_colors", "palette", "GENE_COLOR_COLUMN"]

#: ``var`` column holding a hex colour per gene.
GENE_COLOR_COLUMN = "color"

#: Successive hues are separated by the golden ratio, which keeps neighbouring entries
#: visually distinct instead of walking through a smooth ramp where adjacent genes look
#: identical.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

_SATURATION = 0.65
_LIGHTNESS = 0.55


def palette(n: int) -> list[str]:
    """``n`` visually distinct hex colours, deterministic in ``n`` and position."""
    colors = []
    for i in range(n):
        hue = (i * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, _LIGHTNESS, _SATURATION)
        colors.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return colors


def add_gene_colors(table: Any, column: str = GENE_COLOR_COLUMN, overwrite: bool = False) -> str | None:
    """Add a hex colour per gene to ``table.var``, in place.

    Colours are assigned by position in ``var``, which is the order a client indexes with
    ``feature_code``. Controls are not in ``var`` and get the client's fallback colour.

    Returns
    -------
    The column name, or ``None`` when one already exists and ``overwrite`` is False.
    """
    if column in table.var and not overwrite:
        return None
    table.var[column] = palette(table.n_vars)
    return column


def add_cluster_colors(table: Any, column: str, overwrite: bool = False) -> str | None:
    """Add ``uns["<column>_colors"]`` for a categorical ``obs`` column, in place.

    Follows the scanpy convention: one colour per category, in category order.

    Returns
    -------
    The ``uns`` key, or ``None`` when the column is missing or not categorical.
    """
    if column not in table.obs:
        return None

    values = table.obs[column]
    categories = getattr(getattr(values, "cat", None), "categories", None)
    if categories is None:
        return None

    key = f"{column}_colors"
    if key in table.uns and not overwrite:
        return None

    table.uns[key] = palette(len(categories))
    return key
