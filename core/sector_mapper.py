"""Sector normalization and BACI sector-mapping helpers."""

from __future__ import annotations

from typing import Optional

import pandas as pd

SUPPORTED_SECTORS = ("all", "energy", "agriculture", "electronics")



def normalize_sector(sector: Optional[str]) -> str:
    """Normalize a sector selection to a supported value."""
    if sector is None:
        return "all"

    normalized = str(sector).strip().lower()
    if not normalized:
        return "all"
    if normalized not in SUPPORTED_SECTORS:
        supported = ", ".join(SUPPORTED_SECTORS)
        raise ValueError(f"Unsupported sector '{sector}'. Supported values: {supported}")
    return normalized



def normalize_sector_column(data: pd.DataFrame, column: str = "sector") -> pd.Series:
    """Normalize a sector column to lowercase labels."""
    return data[column].astype(str).str.strip().str.lower().replace({"": "all", "nan": "all", "none": "all"})



def filter_trade_frame_by_sector(data: pd.DataFrame, sector: Optional[str]) -> pd.DataFrame:
    """Filter a normalized trade frame by sector when requested."""
    selected_sector = normalize_sector(sector)
    if selected_sector == "all":
        return data.copy()

    if "sector" not in data.columns:
        raise ValueError("Trade data does not include a sector column for sector-specific filtering.")

    filtered = data[normalize_sector_column(data) == selected_sector].copy()
    if filtered.empty:
        raise ValueError(f"No trade data is available for sector '{selected_sector}'.")
    return filtered.reset_index(drop=True)



def sector_summary_label(sector: Optional[str]) -> str:
    """Return a human-readable label for sector summaries."""
    selected_sector = normalize_sector(sector)
    if selected_sector == "all":
        return "All sectors"
    return f"{selected_sector.title()} sector"



def baci_sector_mask(product_codes: pd.Series, sector: Optional[str]) -> pd.Series:
    """Return a boolean mask for BACI product codes belonging to a sector.

    Sector mapping uses broad HS chapter groupings:
    - agriculture: chapters 01-24
    - energy: chapter 27
    - electronics: chapters 84, 85, and 90
    """
    selected_sector = normalize_sector(sector)
    if selected_sector == "all":
        return pd.Series(True, index=product_codes.index)

    numeric_codes = pd.to_numeric(product_codes, errors="coerce").fillna(-1).astype(int)
    chapters = numeric_codes // 10000

    if selected_sector == "agriculture":
        return chapters.between(1, 24)
    if selected_sector == "energy":
        return chapters == 27
    if selected_sector == "electronics":
        return chapters.isin([84, 85, 90])

    return pd.Series(False, index=product_codes.index)
