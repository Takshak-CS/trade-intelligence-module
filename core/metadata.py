"""Country economic metadata, and what it lets the trade analysis say.

Raw trade value measures volume. It does not measure consequence. A $10bn
export loss is a rounding error for Germany and an existential event for
Djibouti, and nothing else in this module can tell those two cases apart —
every score it produces is relative to the trade network, never to the economy
underneath it.

GDP and population fix that. With them, a shock impact score becomes a share of
national output, and total trade becomes a trade-to-GDP openness ratio.

The metadata file is optional. When it is absent every function here returns
None or passes frames through untouched, so the module runs exactly as before —
enrichment is additive and never a hard dependency. Populate it with:

    python scripts/fetch_metadata.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

METADATA_FILENAME = "country_metadata.parquet"

REQUIRED_COLUMNS = ("country_iso3", "year", "gdp_usd")


def metadata_path(cache_dir: str | Path) -> Path:
    """Location of the country metadata table inside the cache."""
    return Path(cache_dir) / METADATA_FILENAME


def metadata_available(cache_dir: str | Path) -> bool:
    """Return True when economic metadata can be loaded."""
    return metadata_path(cache_dir).exists()


def load_metadata(cache_dir: str | Path) -> Optional[pd.DataFrame]:
    """Load country economic metadata, or None when it has not been fetched."""
    path = metadata_path(cache_dir)
    if not path.exists():
        return None

    metadata = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(
            f"Country metadata is missing required columns: {', '.join(missing)}. "
            "Rebuild it with scripts/fetch_metadata.py."
        )
    return metadata


def metadata_for_year(
    metadata: Optional[pd.DataFrame],
    year: int,
    max_backfill: int = 5,
) -> Optional[pd.DataFrame]:
    """Select one year of metadata, falling back to the most recent prior year.

    World Bank figures lag, and the newest BACI year often has no GDP published
    yet. Rather than dropping those countries, carry the last known value
    forward up to ``max_backfill`` years and record how stale it is, so callers
    can decide whether to trust it.
    """
    if metadata is None or metadata.empty:
        return None

    window = metadata[
        (metadata["year"] <= int(year)) & (metadata["year"] >= int(year) - int(max_backfill))
    ]
    if window.empty:
        return None

    latest = (
        window.sort_values("year")
        .groupby("country_iso3", as_index=False)
        .last()
    )
    latest["metadata_year"] = latest["year"]
    latest["metadata_lag"] = int(year) - latest["year"]
    return latest.drop(columns=["year"])


def attach_openness(
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    country_codes: Optional[pd.DataFrame],
    year: int,
) -> pd.DataFrame:
    """Add trade-to-GDP openness to a country feature frame.

    Returns the frame unchanged when metadata is unavailable, so callers do not
    need to branch on whether enrichment happened.
    """
    yearly = metadata_for_year(metadata, year)
    if yearly is None or country_codes is None:
        return features

    named = country_codes.loc[:, ["country_name", "country_iso3"]].rename(
        columns={"country_name": "country"}
    )
    enriched = features.merge(named, on="country", how="left").merge(
        yearly, on="country_iso3", how="left"
    )

    gdp = pd.to_numeric(enriched.get("gdp_usd"), errors="coerce")
    # BACI reports trade in thousands of USD; World Bank reports GDP in USD.
    trade_usd = enriched["trade_total"] * 1_000.0
    enriched["trade_to_gdp"] = (trade_usd / gdp).where(gdp > 0)

    if "population" in enriched.columns:
        population = pd.to_numeric(enriched["population"], errors="coerce")
        enriched["trade_per_capita"] = (trade_usd / population).where(population > 0)

    return enriched


def attach_economic_exposure(
    impact: pd.DataFrame,
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame],
    country_codes: Optional[pd.DataFrame],
    year: int,
) -> pd.DataFrame:
    """Translate shock impact scores into a share of national output.

    ``impact_score`` is the fraction of a country's imports lost in the
    simulation. Multiplying it by that country's trade and dividing by GDP
    gives ``gdp_exposure``: roughly how much of the economy the disruption
    touches. That is the number a policy reader actually understands.
    """
    yearly = metadata_for_year(metadata, year)
    if yearly is None or country_codes is None:
        return impact

    named = country_codes.loc[:, ["country_name", "country_iso3"]].rename(
        columns={"country_name": "country"}
    )
    trade_columns = ["country", "total_imports", "trade_total"]
    available = [column for column in trade_columns if column in features.columns]

    enriched = (
        impact.merge(features.loc[:, available], on="country", how="left")
        .merge(named, on="country", how="left")
        .merge(yearly, on="country_iso3", how="left")
    )

    gdp = pd.to_numeric(enriched.get("gdp_usd"), errors="coerce")
    imports_usd = enriched.get("total_imports", pd.Series(dtype=float)) * 1_000.0
    enriched["disrupted_trade_usd"] = enriched["impact_score"] * imports_usd
    enriched["gdp_exposure"] = (enriched["disrupted_trade_usd"] / gdp).where(gdp > 0)

    return enriched


def describe_exposure(gdp_exposure: Optional[float]) -> str:
    """Turn a GDP exposure ratio into a plain-language severity band."""
    if gdp_exposure is None or pd.isna(gdp_exposure):
        return "economic exposure unknown"
    if gdp_exposure >= 0.10:
        return "severe economic exposure"
    if gdp_exposure >= 0.03:
        return "material economic exposure"
    if gdp_exposure >= 0.01:
        return "moderate economic exposure"
    return "limited economic exposure"
