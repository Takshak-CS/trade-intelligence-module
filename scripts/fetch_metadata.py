"""Fetch country GDP and population from the World Bank into the parquet cache.

The trade analysis can rank who is exposed to a disruption, but without knowing
how big each economy is it cannot say who can absorb one. This script fills that
gap by pulling two public World Bank indicators and writing them alongside the
trade cache.

It talks to api.worldbank.org, so it needs network access. It is a one-time
offline step, not something the API server ever does at request time. Everything
in the module works without it — enrichment is additive.

Usage:
    python scripts/fetch_metadata.py
    python scripts/fetch_metadata.py --start 1995 --end 2024
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metadata import metadata_path  # noqa: E402

API_ROOT = "https://api.worldbank.org/v2"
INDICATORS = {
    "gdp_usd": "NY.GDP.MKTP.CD",
    "population": "SP.POP.TOTL",
}
PAGE_SIZE = 20_000
TIMEOUT_SECONDS = 60


def fetch_indicator(indicator: str, start: int, end: int) -> List[dict]:
    """Download one World Bank indicator across all countries and years."""
    url = (
        f"{API_ROOT}/country/all/indicator/{indicator}"
        f"?format=json&per_page={PAGE_SIZE}&date={start}:{end}"
    )

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not reach the World Bank API: {exc}") from exc

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise SystemExit(f"The World Bank API returned no rows for indicator '{indicator}'.")

    return payload[1]


def to_frame(rows: List[dict], value_name: str) -> pd.DataFrame:
    """Reshape raw World Bank rows into an ISO3-keyed table."""
    records = []
    for row in rows:
        iso3 = (row.get("countryiso3code") or "").strip().upper()
        # Aggregates like "World" and "Arab World" carry blank or non-standard
        # codes; only real three-letter country codes are useful for joining.
        if len(iso3) != 3 or row.get("value") is None:
            continue
        records.append(
            {
                "country_iso3": iso3,
                "year": int(row["date"]),
                value_name: float(row["value"]),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch World Bank country metadata.")
    parser.add_argument("--cache", default="cache", help="Cache directory to write into.")
    parser.add_argument("--start", type=int, default=1995, help="First year to fetch.")
    parser.add_argument("--end", type=int, default=2024, help="Last year to fetch.")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    for value_name, indicator in INDICATORS.items():
        print(f"Fetching {value_name} ({indicator}) for {args.start}-{args.end} ...", flush=True)
        frame = to_frame(fetch_indicator(indicator, args.start, args.end), value_name)
        print(f"  {len(frame):,} observations across {frame['country_iso3'].nunique()} countries", flush=True)
        frames[value_name] = frame

    merged = frames["gdp_usd"]
    for value_name, frame in frames.items():
        if value_name == "gdp_usd":
            continue
        merged = merged.merge(frame, on=["country_iso3", "year"], how="outer")

    merged = merged.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    target = metadata_path(cache_dir)
    merged.to_parquet(target, index=False, compression="snappy")

    print(
        f"\nWrote {len(merged):,} rows covering {merged['country_iso3'].nunique()} countries "
        f"to {target.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
