"""Read access to the precomputed parquet cache.

The cache is produced by ``scripts/build_cache.py`` and holds the same country
level data the raw BACI CSVs would yield, already aggregated. When it is present
the loaders read from here; when it is absent everything falls back to parsing
the CSVs directly, so the cache is an optimisation and never a requirement.

The cache is also self-contained: it carries its own copy of the country code
table, so a teammate who has the cache but not the 8 GB dataset can still run
the whole module.
"""

from __future__ import annotations

import difflib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = "cache"
MAX_CACHED_YEARS = 8

_MEMORY_LOCK = threading.Lock()
_MEMORY: "OrderedDict[tuple[str, int, str], pd.DataFrame]" = OrderedDict()


def edges_path(cache_dir: Path, year: int) -> Path:
    """Path to the aggregated edge table for one year."""
    return Path(cache_dir) / "edges" / f"edges_{int(year)}.parquet"


def country_year_path(cache_dir: Path) -> Path:
    """Path to the per-country yearly totals table."""
    return Path(cache_dir) / "country_year.parquet"


def quality_path(cache_dir: Path) -> Path:
    """Path to the data completeness table."""
    return Path(cache_dir) / "quality.parquet"


def countries_path(cache_dir: Path) -> Path:
    """Path to the bundled country code table."""
    return Path(cache_dir) / "countries.parquet"


def centrality_path(cache_dir: Path) -> Path:
    """Path to the precomputed centrality table."""
    return Path(cache_dir) / "centrality.parquet"


def meta_path(cache_dir: Path) -> Path:
    """Path to the build manifest."""
    return Path(cache_dir) / "meta.json"


def cache_available(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    """Return True when a usable cache exists at this location."""
    directory = Path(cache_dir)
    if not meta_path(directory).exists() or not country_year_path(directory).exists():
        return False
    return bool(cached_years(directory))


def cache_meta(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict:
    """Read the build manifest, or an empty dict when there is no cache."""
    path = meta_path(Path(cache_dir))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def cached_years(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[int]:
    """List the years present in the cache."""
    edges_dir = Path(cache_dir) / "edges"
    if not edges_dir.is_dir():
        return []

    years = []
    for path in edges_dir.glob("edges_*.parquet"):
        try:
            years.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(years)


def latest_cached_year(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> int:
    """Return the most recent year in the cache."""
    years = cached_years(cache_dir)
    if not years:
        raise ValueError(f"No cached years were found in '{cache_dir}'.")
    return max(years)


def load_year(
    cache_dir: str | Path,
    year: int,
    sector: str = "all",
) -> pd.DataFrame:
    """Load one cached year, filtered to a sector.

    Results are memoised behind a lock because FastAPI serves sync endpoints
    from a threadpool, and two concurrent first-hits on the same year would
    otherwise both pay the read.
    """
    directory = Path(cache_dir)
    key = (str(directory.resolve()), int(year), str(sector))

    with _MEMORY_LOCK:
        hit = _MEMORY.get(key)
        if hit is not None:
            _MEMORY.move_to_end(key)
            return _copy_with_attrs(hit)

    path = edges_path(directory, year)
    if not path.exists():
        raise ValueError(f"Year {int(year)} is not present in the cache at '{directory}'.")

    frame = pd.read_parquet(path)
    frame = frame[frame["sector"] == str(sector)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No cached trade data for year {int(year)} and sector '{sector}'.")

    frame = frame.loc[:, ["exporter", "importer", "year", "trade_value", "sector"]]
    frame.attrs["data_quality"] = load_quality(directory, year=int(year), sector=str(sector))
    frame.attrs["sector"] = str(sector)

    with _MEMORY_LOCK:
        _MEMORY[key] = frame
        _MEMORY.move_to_end(key)
        while len(_MEMORY) > MAX_CACHED_YEARS:
            _MEMORY.popitem(last=False)

    return _copy_with_attrs(frame)


def load_country_series(
    cache_dir: str | Path,
    country: str,
    sector: str = "all",
) -> pd.DataFrame:
    """Load the full yearly history for one country from the cache."""
    directory = Path(cache_dir)
    totals = pd.read_parquet(country_year_path(directory))
    selected = totals[(totals["country"] == str(country)) & (totals["sector"] == str(sector))]

    if selected.empty:
        raise ValueError(f"No cached trade history for country '{country}' and sector '{sector}'.")

    columns = [
        "year",
        "imports",
        "exports",
        "trade_total",
        "import_dependency_ratio",
        "export_dependency_ratio",
    ]
    return selected.loc[:, columns].sort_values("year").reset_index(drop=True)


def load_quality(
    cache_dir: str | Path,
    year: Optional[int] = None,
    sector: str = "all",
) -> dict:
    """Load the completeness record for a year and sector."""
    path = quality_path(Path(cache_dir))
    if not path.exists():
        return {}

    quality = pd.read_parquet(path)
    selected = quality[quality["sector"] == str(sector)]
    if year is not None:
        selected = selected[selected["year"] == int(year)]

    if selected.empty:
        return {}

    if year is None:
        # Aggregate across years by weighting each year's ratios by its row count.
        weights = selected["row_count"].to_numpy(dtype=float)
        total = float(weights.sum())
        if total <= 0:
            return {}
        return {
            "required_field_availability": float(
                (selected["required_field_availability"].to_numpy(dtype=float) * weights).sum() / total
            ),
            "value_availability": float(
                (selected["value_availability"].to_numpy(dtype=float) * weights).sum() / total
            ),
            "quantity_availability": float(
                (selected["quantity_availability"].to_numpy(dtype=float) * weights).sum() / total
            ),
            "row_count": int(total),
            "quantity_observed": True,
        }

    row = selected.iloc[0]
    return {
        "required_field_availability": float(row["required_field_availability"]),
        "value_availability": float(row["value_availability"]),
        "quantity_availability": float(row["quantity_availability"]),
        "row_count": int(row["row_count"]),
        "quantity_observed": bool(row["quantity_observed"]),
    }


def load_centrality(
    cache_dir: str | Path,
    year: int,
    sector: str = "all",
) -> Optional[pd.DataFrame]:
    """Load precomputed centrality for a year and sector.

    Returns None when the cache predates centrality precomputation or was built
    with it skipped, in which case callers fall back to computing it live.
    """
    path = centrality_path(Path(cache_dir))
    if not path.exists():
        return None

    centrality = pd.read_parquet(path)
    selected = centrality[
        (centrality["year"] == int(year)) & (centrality["sector"] == str(sector))
    ]
    if selected.empty:
        return None

    return selected.loc[
        :, ["country", "pagerank_centrality", "betweenness_centrality"]
    ].reset_index(drop=True)


def load_countries(cache_dir: str | Path) -> pd.DataFrame:
    """Load the bundled country code table."""
    path = countries_path(Path(cache_dir))
    if not path.exists():
        raise ValueError(f"No bundled country table was found in '{cache_dir}'.")
    return pd.read_parquet(path)


# BACI labels countries with formal or abbreviated UN names, several of which
# nobody types from memory: "USA" rather than "United States", "Rep. of Korea"
# rather than "South Korea", "Türkiye" rather than "Turkey". An orchestrator
# passing ordinary country names would fail on all of these, so common usage is
# mapped onto ISO3, which the code table already carries.
COUNTRY_ALIASES: dict[str, str] = {
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    "u.s.": "USA",
    "u.s.a.": "USA",
    "america": "USA",
    "south korea": "KOR",
    "korea": "KOR",
    "republic of korea": "KOR",
    "north korea": "PRK",
    "democratic people's republic of korea": "PRK",
    "russia": "RUS",
    "vietnam": "VNM",
    "turkey": "TUR",
    "turkiye": "TUR",
    "hong kong": "HKG",
    "macao": "MAC",
    "macau": "MAC",
    # BACI follows UN Comtrade, which reports Taiwan's trade under the residual
    # category "Other Asia, nes" rather than as a separate reporter. That
    # category is overwhelmingly Taiwan in practice and is how trade economists
    # read it, but it is a residual and not an exact match - worth stating if a
    # result turns on it.
    "taiwan": "S19",
    "chinese taipei": "S19",
    "laos": "LAO",
    "bolivia": "BOL",
    "moldova": "MDA",
    "tanzania": "TZA",
    "brunei": "BRN",
    "czech republic": "CZE",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "dr congo": "COD",
    "drc": "COD",
    "democratic republic of the congo": "COD",
    "congo-kinshasa": "COD",
    "congo-brazzaville": "COG",
    "republic of the congo": "COG",
    "iran": "IRN",
    "syria": "SYR",
    "venezuela": "VEN",
    "uk": "GBR",
    "great britain": "GBR",
    "britain": "GBR",
    "england": "GBR",
    "uae": "ARE",
    "emirates": "ARE",
    "burma": "MMR",
    "cape verde": "CPV",
    "swaziland": "SWZ",
    "east timor": "TLS",
    "macedonia": "MKD",
    "north macedonia": "MKD",
}


def resolve_country(cache_dir: str | Path, country: str) -> tuple[int, str]:
    """Resolve a country name, ISO2, ISO3, alias, or numeric code against the cache."""
    codes = load_countries(cache_dir)
    text = str(country).strip()

    if text.isdigit():
        matched = codes[codes["country_code"] == int(text)]
    else:
        matched = codes[codes["country_name"].str.casefold() == text.casefold()]
        if matched.empty:
            matched = codes[codes["country_iso3"].str.upper() == text.upper()]
        if matched.empty:
            matched = codes[codes["country_iso2"].str.upper() == text.upper()]
        if matched.empty:
            alias = COUNTRY_ALIASES.get(text.casefold())
            if alias:
                matched = codes[codes["country_iso3"].str.upper() == alias]

    if matched.empty:
        raise ValueError(country_not_found_message(codes, text))

    row = matched.iloc[0]
    return int(row["country_code"]), str(row["country_name"])


def country_not_found_message(codes: pd.DataFrame, text: str) -> str:
    """Build an error that helps the caller find the right identifier.

    Two traps make a bare "not found" unhelpful here. BACI numeric codes are
    not ISO 3166 numeric codes for every country (India is 699, not 356), so
    the obvious guess fails silently. And several BACI names carry qualifiers
    ("China, Hong Kong SAR", "Other Asia, nes") that nobody types from memory.
    Suggesting near matches turns a dead end into a next step.
    """
    if str(text).strip().isdigit():
        return (
            f"Country code '{text}' was not found. Note that BACI uses its own numeric "
            f"codes, which differ from ISO 3166 numeric for many countries. "
            f"An ISO2, ISO3, or country name is usually the safer identifier."
        )

    candidates = difflib.get_close_matches(
        str(text).casefold(),
        [name.casefold() for name in codes["country_name"]],
        n=3,
        cutoff=0.6,
    )
    lookup = {name.casefold(): name for name in codes["country_name"]}
    suggestions = [lookup[candidate] for candidate in candidates if candidate in lookup]

    if not suggestions:
        # Fall back to substring matching, which catches qualified names like
        # "Hong Kong" -> "China, Hong Kong SAR" that fuzzy ratio misses.
        contains = codes.loc[
            codes["country_name"].str.contains(str(text).strip(), case=False, regex=False),
            "country_name",
        ]
        suggestions = contains.head(3).tolist()

    if suggestions:
        return f"Country '{text}' was not found. Did you mean: {', '.join(suggestions)}?"
    return (
        f"Country '{text}' was not found in the BACI country table. "
        f"Accepted identifiers are country name, ISO2, ISO3, or BACI numeric code."
    )


def known_countries(cache_dir: str | Path, sector: str = "all") -> list[str]:
    """List every country that appears in the cached totals for a sector."""
    totals = pd.read_parquet(country_year_path(Path(cache_dir)), columns=["country", "sector"])
    return sorted(totals.loc[totals["sector"] == str(sector), "country"].unique().tolist())


def clear_memory() -> None:
    """Drop the in-process year cache. Used by tests."""
    with _MEMORY_LOCK:
        _MEMORY.clear()


def _copy_with_attrs(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a cached frame while preserving the metadata attached to it."""
    copied = frame.copy(deep=True)
    copied.attrs = dict(frame.attrs)
    return copied
