"""Assign display colours to genes and to categorical cell annotations.

There is no biologically correct colour for a gene, so these are invented either way. The
question is only where they live, and AnnData already has an answer: ``uns`` holds
``<name>_colors`` lists aligned to an ordering. Both functions here follow it, so nothing
new is being proposed -- gene colours go to ``uns["gene_colors"]`` in ``var_names`` order,
cluster colours to ``uns["<column>_colors"]`` in category order, exactly as scanpy writes
them.

This is better than a ``var`` column for the same reason it is better than a viewer-specific
file: it is the shape existing tools already look for.

The palette matches the fallback Celldega generates when a store has no colours, so a store
looks the same whether or not this ran.
"""

from __future__ import annotations

import colorsys
from typing import Any

__all__ = ["add_gene_colors", "add_cluster_colors", "palette", "GENE_COLORS_KEY"]

#: ``uns`` key holding one hex colour per gene, in ``var_names`` order. Follows AnnData's
#: existing ``<name>_colors`` convention rather than introducing a ``var`` column.
GENE_COLORS_KEY = "gene_colors"

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


def add_gene_colors(table: Any, key: str = GENE_COLORS_KEY, overwrite: bool = False) -> str | None:
    """Add one hex colour per gene to ``table.uns``, in place.

    Colours are ordered by position in ``var``, which is the order a client indexes with
    ``feature_code``. Controls are not in ``var`` and get the client's fallback colour.

    Returns
    -------
    The ``uns`` key, or ``None`` when one already exists and ``overwrite`` is False.
    """
    if key in table.uns and not overwrite:
        return None
    table.uns[key] = palette(table.n_vars)
    return key


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
