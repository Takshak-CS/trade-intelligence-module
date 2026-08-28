"""Repair double-encoded country names in an existing parquet cache.

The CEPII country code file ships double-encoded UTF-8, so caches built before
that was handled carry mangled names for four countries: Côte d'Ivoire,
Curaçao, Saint Barthélemy, and Türkiye. Country names are the only thing the
build derives from that table, so the damage is a pure string substitution and
does not need an 8 GB rebuild to undo.

This rewrites the affected names in every cached table. It is idempotent: run
it twice and the second run reports nothing to do.

Usage:
    python scripts/repair_cache_encoding.py
    python scripts/repair_cache_encoding.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.baci_cache import (  # noqa: E402
    cached_years,
    centrality_path,
    countries_path,
    country_year_path,
    edges_path,
)
from core.data_loader import repair_mojibake  # noqa: E402

# Which columns hold country names in each cached table.
NAME_COLUMNS = {
    "countries.parquet": ["country_name"],
    "country_year.parquet": ["country"],
    "centrality.parquet": ["country"],
    "edges": ["exporter", "importer"],
}


def build_rename_map(cache_dir: Path) -> Dict[str, str]:
    """Find every cached country name that needs repairing."""
    countries = pd.read_parquet(countries_path(cache_dir))
    mapping = {}
    for name in countries["country_name"].astype(str):
        repaired = repair_mojibake(name)
        if repaired != name:
            mapping[name] = repaired
    return mapping


def rewrite(path: Path, columns: List[str], mapping: Dict[str, str], dry_run: bool) -> int:
    """Apply the rename to one parquet file, returning how many cells changed."""
    if not path.exists():
        return 0

    frame = pd.read_parquet(path)
    changed = 0
    for column in columns:
        if column not in frame.columns:
            continue
        hits = int(frame[column].isin(mapping).sum())
        if hits:
            changed += hits
            frame[column] = frame[column].replace(mapping)

    if changed and not dry_run:
        frame.to_parquet(path, index=False, compression="snappy")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair country name encoding in the cache.")
    parser.add_argument("--cache", default="cache", help="Cache directory to repair.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    if not countries_path(cache_dir).exists():
        raise SystemExit(f"No country table found in '{cache_dir}'. Nothing to repair.")

    mapping = build_rename_map(cache_dir)
    if not mapping:
        print("Country names are already correctly encoded. Nothing to do.")
        return

    print(f"Repairing {len(mapping)} country names:")
    for bad, good in mapping.items():
        print(f"  {bad!r} -> {good!r}")
    print()

    total = 0
    for filename, columns in NAME_COLUMNS.items():
        if filename == "edges":
            continue
        changed = rewrite(cache_dir / filename, columns, mapping, args.dry_run)
        total += changed
        print(f"  {filename:<24} {changed:>8,} cells")

    for year in cached_years(cache_dir):
        changed = rewrite(edges_path(cache_dir, year), NAME_COLUMNS["edges"], mapping, args.dry_run)
        total += changed
        if changed:
            print(f"  edges_{year}.parquet{'':<8} {changed:>8,} cells")

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {total:,} cells across the cache.")


if __name__ == "__main__":
    main()
