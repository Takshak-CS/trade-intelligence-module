"""Utilities for loading and validating trade data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from core.sector_mapper import (
    baci_sector_mask,
    filter_trade_frame_by_sector,
    normalize_sector,
    normalize_sector_column,
)

REQUIRED_COLUMNS = ("exporter", "importer", "year", "trade_value")
BACI_FILE_GLOB = "BACI_HS92_Y*_V*.csv"
BACI_CHUNKSIZE = 250_000



def load_trade_data(
    csv_path: str | Path,
    year: Optional[int] = None,
    sector: Optional[str] = "all",
) -> pd.DataFrame:
    """Load trade data from a normalized CSV or a BACI dataset directory."""
    source = Path(csv_path)
    selected_sector = normalize_sector(sector)
    if is_baci_directory(source):
        selected_year = latest_year(source) if year is None else int(year)
        return load_baci_year(source, selected_year, sector=selected_sector)

    raw_data = pd.read_csv(source)
    quality_metadata = _summarize_normalized_quality(raw_data, year=year, sector=selected_sector)
    cleaned = clean_trade_data(raw_data)
    filtered = filter_by_year(cleaned, year)
    filtered = filter_trade_frame_by_sector(filtered, selected_sector)
    filtered.attrs["data_quality"] = quality_metadata
    filtered.attrs["sector"] = selected_sector
    return filtered



def load_country_time_series(
    csv_path: str | Path,
    country: str,
    sector: Optional[str] = "all",
) -> dict:
    """Load a yearly country time series from a normalized CSV or BACI directory."""
    source = Path(csv_path)
    selected_sector = normalize_sector(sector)
    if is_baci_directory(source):
        return load_baci_country_time_series(source, country, sector=selected_sector)

    data = load_trade_data(source, sector=selected_sector)
    return {
        "country": str(country),
        "sector": selected_sector,
        "data_quality": data.attrs.get("data_quality", {}),
        "time_series": _build_country_time_series_from_frame(data, str(country)),
    }



def clean_trade_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, enforce schema, and drop invalid rows."""
    normalized = data.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    validate_trade_columns(normalized)

    if "sector" not in normalized.columns:
        normalized["sector"] = "all"

    normalized = normalized.loc[:, (*REQUIRED_COLUMNS, "sector")].copy()
    normalized["exporter"] = normalized["exporter"].astype(str).str.strip()
    normalized["importer"] = normalized["importer"].astype(str).str.strip()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    normalized["trade_value"] = pd.to_numeric(normalized["trade_value"], errors="coerce")
    normalized["sector"] = normalize_sector_column(normalized)

    normalized["exporter"] = normalized["exporter"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    normalized["importer"] = normalized["importer"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    normalized = normalized.dropna(subset=REQUIRED_COLUMNS)

    normalized = normalized[normalized["trade_value"] >= 0].copy()
    normalized = normalized[normalized["year"] > 0].copy()
    normalized["year"] = normalized["year"].astype(int)
    normalized = normalized[normalized["exporter"] != normalized["importer"]].copy()

    if normalized.empty:
        raise ValueError("Trade dataset contains no valid rows after cleaning.")

    normalized = normalized.sort_values(["year", "sector", "exporter", "importer"]).reset_index(drop=True)
    return normalized



def validate_trade_columns(data: pd.DataFrame) -> None:
    """Raise an error when the required trade columns are missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Trade dataset is missing required columns: {missing_str}")



def filter_by_year(data: pd.DataFrame, year: Optional[int]) -> pd.DataFrame:
    """Return the full dataset or the subset for a specific year."""
    if year is None:
        return data.copy()

    filtered = data[data["year"] == int(year)].copy()
    if filtered.empty:
        raise ValueError(f"No trade data is available for year {int(year)}.")
    return filtered.reset_index(drop=True)



def latest_year(data_or_source: pd.DataFrame | str | Path) -> int:
    """Return the most recent year available in a dataframe or data source."""
    if isinstance(data_or_source, pd.DataFrame):
        if data_or_source.empty:
            raise ValueError("Cannot determine the latest year from an empty dataset.")
        return int(data_or_source["year"].max())

    source = Path(data_or_source)
    if is_baci_directory(source):
        years = available_years(source)
        if not years:
            raise ValueError(f"No BACI yearly files were found in '{source}'.")
        return max(years)

    data = pd.read_csv(source, usecols=["year"])
    if data.empty:
        raise ValueError("Cannot determine the latest year from an empty dataset.")
    return int(pd.to_numeric(data["year"], errors="coerce").dropna().max())



def is_baci_directory(source: str | Path) -> bool:
    """Return True when the source is a BACI directory with yearly files."""
    path = Path(source)
    return path.is_dir() and any(path.glob(BACI_FILE_GLOB))



def available_years(source: str | Path) -> list[int]:
    """List available BACI years from the directory naming pattern."""
    path = Path(source)
    return sorted(_extract_year_from_filename(file_path.name) for file_path in path.glob(BACI_FILE_GLOB))



def load_baci_year(directory: str | Path, year: int, sector: Optional[str] = "all") -> pd.DataFrame:
    """Load and aggregate a single BACI year to country-country trade flows."""
    dataset_dir = Path(directory)
    selected_sector = normalize_sector(sector)
    file_path = _find_baci_file(dataset_dir, int(year))
    partial_frames = []
    quality_stats = _quality_stats_template()
    usecols = ["t", "i", "j", "v", "q"] if selected_sector == "all" else ["t", "i", "j", "k", "v", "q"]
    dtypes = {"t": "int16", "i": "int32", "j": "int32", "v": "float64", "q": "float64"}
    if selected_sector != "all":
        dtypes["k"] = "int32"

    for chunk in pd.read_csv(file_path, usecols=usecols, dtype=dtypes, chunksize=BACI_CHUNKSIZE):
        working = chunk
        if selected_sector != "all":
            working = chunk.loc[baci_sector_mask(chunk["k"], selected_sector), ["t", "i", "j", "v", "q"]]
            if working.empty:
                continue

        _update_quality_stats(quality_stats, working, required_columns=["t", "i", "j", "v"], quantity_column="q")
        partial_frames.append(working.groupby(["t", "i", "j"], as_index=False)["v"].sum())

    if not partial_frames:
        raise ValueError(
            f"No BACI rows were loaded for year {int(year)} and sector '{selected_sector}'."
        )

    aggregated = (
        pd.concat(partial_frames, ignore_index=True)
        .groupby(["t", "i", "j"], as_index=False)["v"]
        .sum()
    )

    country_codes = _load_country_codes_cached(str(dataset_dir))
    country_map = country_codes.set_index("country_code")["country_name"].to_dict()
    aggregated["exporter"] = aggregated["i"].map(country_map)
    aggregated["importer"] = aggregated["j"].map(country_map)
    aggregated["sector"] = selected_sector

    normalized = aggregated.rename(columns={"t": "year", "v": "trade_value"})[
        ["exporter", "importer", "year", "trade_value", "sector"]
    ]
    cleaned = clean_trade_data(normalized)
    cleaned.attrs["data_quality"] = _finalize_quality_stats(quality_stats)
    cleaned.attrs["sector"] = selected_sector
    return cleaned



def load_baci_country_time_series(
    directory: str | Path,
    country: str,
    sector: Optional[str] = "all",
) -> dict:
    """Build a country time series by scanning BACI yearly files on demand."""
    dataset_dir = Path(directory)
    selected_sector = normalize_sector(sector)
    country_code, country_name = resolve_baci_country(dataset_dir, country)
    records = []
    quality_stats = _quality_stats_template()

    for year in available_years(dataset_dir):
        file_path = _find_baci_file(dataset_dir, year)
        total_imports = 0.0
        total_exports = 0.0
        usecols = ["i", "j", "v", "q"] if selected_sector == "all" else ["i", "j", "k", "v", "q"]
        dtypes = {"i": "int32", "j": "int32", "v": "float64", "q": "float64"}
        if selected_sector != "all":
            dtypes["k"] = "int32"

        for chunk in pd.read_csv(file_path, usecols=usecols, dtype=dtypes, chunksize=BACI_CHUNKSIZE):
            working = chunk
            if selected_sector != "all":
                working = chunk.loc[baci_sector_mask(chunk["k"], selected_sector), ["i", "j", "v", "q"]]
                if working.empty:
                    continue

            relevant = working[(working["i"] == country_code) | (working["j"] == country_code)].copy()
            if relevant.empty:
                continue

            _update_quality_stats(quality_stats, relevant, required_columns=["i", "j", "v"], quantity_column="q")
            total_exports += float(relevant.loc[relevant["i"] == country_code, "v"].sum())
            total_imports += float(relevant.loc[relevant["j"] == country_code, "v"].sum())

        trade_total = total_imports + total_exports
        records.append(
            {
                "year": int(year),
                "imports": float(total_imports),
                "exports": float(total_exports),
                "trade_total": float(trade_total),
                "import_dependency_ratio": float(total_imports / trade_total) if trade_total else 0.0,
                "export_dependency_ratio": float(total_exports / trade_total) if trade_total else 0.0,
            }
        )

    time_series = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    if time_series.empty:
        raise ValueError(
            f"No BACI time series could be built for country '{country_name}' and sector '{selected_sector}'."
        )

    return {
        "country": country_name,
        "sector": selected_sector,
        "data_quality": _finalize_quality_stats(quality_stats),
        "time_series": time_series,
    }



def resolve_baci_country(directory: str | Path, country: str) -> tuple[int, str]:
    """Resolve a country name, ISO code, or numeric code to a BACI country code."""
    dataset_dir = Path(directory)
    country_codes = _load_country_codes_cached(str(dataset_dir)).copy()
    country_text = str(country).strip()

    if country_text.isdigit():
        matched = country_codes[country_codes["country_code"] == int(country_text)]
    else:
        upper_text = country_text.upper()
        matched = country_codes[country_codes["country_name"].str.casefold() == country_text.casefold()]
        if matched.empty:
            matched = country_codes[country_codes["country_iso3"].str.upper() == upper_text]
        if matched.empty:
            matched = country_codes[country_codes["country_iso2"].str.upper() == upper_text]

    if matched.empty:
        raise ValueError(f"Country '{country}' was not found in BACI country codes.")

    row = matched.iloc[0]
    return int(row["country_code"]), str(row["country_name"])



def _build_country_time_series_from_frame(data: pd.DataFrame, country: str) -> pd.DataFrame:
    """Aggregate yearly totals for a country from a normalized trade frame."""
    relevant = data[(data["exporter"] == country) | (data["importer"] == country)].copy()
    if relevant.empty:
        raise ValueError(f"No trade history is available for country '{country}'.")

    records = []
    for year, year_frame in relevant.groupby("year"):
        total_imports = float(year_frame.loc[year_frame["importer"] == country, "trade_value"].sum())
        total_exports = float(year_frame.loc[year_frame["exporter"] == country, "trade_value"].sum())
        trade_total = total_imports + total_exports
        records.append(
            {
                "year": int(year),
                "imports": total_imports,
                "exports": total_exports,
                "trade_total": trade_total,
                "import_dependency_ratio": total_imports / trade_total if trade_total else 0.0,
                "export_dependency_ratio": total_exports / trade_total if trade_total else 0.0,
            }
        )

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)



def _find_baci_file(directory: Path, year: int) -> Path:
    """Locate the BACI CSV file for a specific year."""
    matches = sorted(directory.glob(f"BACI_HS92_Y{int(year)}_V*.csv"))
    if not matches:
        raise ValueError(f"No BACI file was found for year {int(year)} in '{directory}'.")
    return matches[0]



def _extract_year_from_filename(file_name: str) -> int:
    """Extract the year token from a BACI file name."""
    year_start = file_name.index("_Y") + 2
    return int(file_name[year_start:year_start + 4])



def _summarize_normalized_quality(data: pd.DataFrame, year: Optional[int], sector: str) -> dict:
    """Summarize completeness for a normalized CSV source."""
    normalized = data.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if not all(column in normalized.columns for column in REQUIRED_COLUMNS):
        return {}

    if "sector" not in normalized.columns:
        normalized["sector"] = "all"

    if year is not None:
        year_series = pd.to_numeric(normalized["year"], errors="coerce")
        normalized = normalized[year_series == int(year)].copy()
    if sector != "all":
        normalized = normalized[normalize_sector_column(normalized) == sector].copy()

    quantity_column = _find_quantity_column(normalized.columns)
    stats = _quality_stats_template()
    _update_quality_stats(stats, normalized, required_columns=list(REQUIRED_COLUMNS), quantity_column=quantity_column)
    return _finalize_quality_stats(stats)



def _find_quantity_column(columns: pd.Index) -> Optional[str]:
    """Find a likely quantity column in a normalized dataset."""
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in ("quantity", "trade_quantity", "q"):
        if candidate in lowered:
            return lowered[candidate]
    return None



def _quality_stats_template() -> dict:
    """Create an empty quality statistics accumulator."""
    return {
        "row_count": 0,
        "required_complete_count": 0,
        "value_present_count": 0,
        "quantity_present_count": 0,
        "quantity_observed": False,
    }



def _update_quality_stats(
    stats: dict,
    frame: pd.DataFrame,
    required_columns: list[str],
    quantity_column: Optional[str],
) -> None:
    """Update quality statistics from a raw frame slice."""
    stats["row_count"] += len(frame)
    if frame.empty:
        return

    available_required = [column for column in required_columns if column in frame.columns]
    if available_required:
        stats["required_complete_count"] += int(frame[available_required].notna().all(axis=1).sum())

    if "v" in frame.columns:
        stats["value_present_count"] += int(frame["v"].notna().sum())
    elif "trade_value" in frame.columns:
        stats["value_present_count"] += int(frame["trade_value"].notna().sum())

    if quantity_column and quantity_column in frame.columns:
        stats["quantity_observed"] = True
        stats["quantity_present_count"] += int(frame[quantity_column].notna().sum())



def _finalize_quality_stats(stats: dict) -> dict:
    """Convert accumulated quality counts into normalized ratios."""
    row_count = int(stats.get("row_count", 0))
    if row_count <= 0:
        return {
            "required_field_availability": 0.0,
            "value_availability": 0.0,
            "quantity_availability": 0.0,
            "row_count": 0,
            "quantity_observed": bool(stats.get("quantity_observed", False)),
        }

    quantity_observed = bool(stats.get("quantity_observed", False))
    quantity_availability = (
        float(stats.get("quantity_present_count", 0)) / row_count if quantity_observed else None
    )
    return {
        "required_field_availability": float(stats.get("required_complete_count", 0)) / row_count,
        "value_availability": float(stats.get("value_present_count", 0)) / row_count,
        "quantity_availability": quantity_availability,
        "row_count": row_count,
        "quantity_observed": quantity_observed,
    }



@lru_cache(maxsize=8)
def _load_country_codes_cached(directory: str) -> pd.DataFrame:
    """Load BACI country code mappings for a dataset directory."""
    dataset_dir = Path(directory)
    matches = sorted(dataset_dir.glob("country_codes_V*.csv"))
    if not matches:
        raise ValueError(f"No BACI country code file was found in '{dataset_dir}'.")

    country_codes = pd.read_csv(
        matches[0],
        usecols=["country_code", "country_name", "country_iso2", "country_iso3"],
    )
    country_codes["country_name"] = country_codes["country_name"].astype(str).str.strip()
    country_codes["country_iso2"] = country_codes["country_iso2"].astype(str).str.strip()
    country_codes["country_iso3"] = country_codes["country_iso3"].astype(str).str.strip()
    return country_codes
