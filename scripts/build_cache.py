"""Precompute a fast parquet cache from the raw BACI yearly CSV files.

The raw dataset is roughly 8 GB spread across thirty yearly CSVs of about ten
million product-level rows each. Every analytical query in this project works at
the country-to-country level, so re-parsing those CSVs per request is wasted
work: a cold forecast has to touch all thirty files and takes over five minutes.

This script makes one pass over each yearly file and writes two derived tables:

    cache/edges/edges_<year>.parquet   aggregated exporter-importer flows
    cache/country_year.parquet         yearly totals per country
    cache/quality.parquet              data completeness per year and sector
    cache/meta.json                    build manifest

Together they come to a few tens of megabytes, which is small enough to commit
or attach to a release, and they turn every query in the module into a filter
over a small table.

Usage:
    python -m scripts.build_cache
    python -m scripts.build_cache --years 2019 2020 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.baci_cache import (  # noqa: E402
    CACHE_VERSION,
    centrality_path,
    countries_path,
    country_year_path,
    edges_path,
    meta_path,
    quality_path,
)
from core.data_loader import (  # noqa: E402
    _find_baci_file,
    _load_country_codes_cached,
    available_years,
    is_baci_directory,
)
from core.sector_mapper import SUPPORTED_SECTORS  # noqa: E402

READ_CHUNKSIZE = 500_000
SPECIFIC_SECTORS = tuple(sector for sector in SUPPORTED_SECTORS if sector != "all")


def label_sectors(chapters: pd.Series) -> pd.Series:
    """Map HS92 chapter numbers onto the sector labels this project reports on.

    The three named sectors use disjoint chapter ranges, so every product row
    gets exactly one label. Rows outside all three are labelled ``other`` and
    still count toward the ``all`` sector totals.
    """
    labels = pd.Series("other", index=chapters.index, dtype="object")
    labels[chapters.between(1, 24)] = "agriculture"
    labels[chapters == 27] = "energy"
    labels[chapters.isin([84, 85, 90])] = "electronics"
    return labels


def build_year(
    dataset_dir: Path,
    year: int,
    country_map: Dict[int, str],
) -> tuple[pd.DataFrame, List[dict]]:
    """Aggregate one BACI year into country-level edges for every sector."""
    file_path = _find_baci_file(dataset_dir, year)

    partials: List[pd.DataFrame] = []
    row_count = 0
    required_complete = 0
    value_present = 0
    quantity_present = 0

    reader = pd.read_csv(
        file_path,
        usecols=["t", "i", "j", "k", "v", "q"],
        dtype={"t": "int16", "i": "int32", "j": "int32", "k": "int32", "v": "float64", "q": "float64"},
        chunksize=READ_CHUNKSIZE,
    )

    for chunk in reader:
        row_count += len(chunk)
        required_complete += int(chunk[["t", "i", "j", "v"]].notna().all(axis=1).sum())
        value_present += int(chunk["v"].notna().sum())
        quantity_present += int(chunk["q"].notna().sum())

        chunk = chunk.assign(sector=label_sectors(chunk["k"] // 10000))
        partials.append(chunk.groupby(["i", "j", "sector"], as_index=False, observed=True)["v"].sum())

    if not partials:
        raise ValueError(f"No rows were read for year {year}.")

    labelled = (
        pd.concat(partials, ignore_index=True)
        .groupby(["i", "j", "sector"], as_index=False, observed=True)["v"]
        .sum()
    )

    # "all" is every product regardless of sector, so it is the sum over labels
    # rather than a fourth pass over the file.
    everything = labelled.groupby(["i", "j"], as_index=False)["v"].sum()
    everything["sector"] = "all"

    combined = pd.concat([labelled[labelled["sector"] != "other"], everything], ignore_index=True)
    combined["exporter"] = combined["i"].map(country_map)
    combined["importer"] = combined["j"].map(country_map)
    combined["year"] = int(year)

    edges = (
        combined.dropna(subset=["exporter", "importer"])
        .rename(columns={"v": "trade_value"})
        .loc[:, ["exporter", "importer", "year", "sector", "trade_value"]]
    )
    edges = edges[edges["exporter"] != edges["importer"]]
    edges = edges[edges["trade_value"] >= 0]
    edges = edges.sort_values(["sector", "exporter", "importer"]).reset_index(drop=True)

    # Completeness is measured once on the raw file, then attributed to each
    # sector slice carved out of it.
    quality_rows = [
        {
            "year": int(year),
            "sector": sector,
            "required_field_availability": required_complete / row_count if row_count else 0.0,
            "value_availability": value_present / row_count if row_count else 0.0,
            "quantity_availability": quantity_present / row_count if row_count else 0.0,
            "row_count": int(row_count),
            "quantity_observed": True,
            "aggregated_edges": int((edges["sector"] == sector).sum()),
        }
        for sector in SUPPORTED_SECTORS
    ]

    return edges, quality_rows


def compute_centrality(edges: pd.DataFrame) -> pd.DataFrame:
    """Precompute PageRank and betweenness for every sector in one year.

    Both metrics are deterministic for a given graph, and weighted betweenness
    is by far the most expensive thing a risk query does — several seconds per
    sector. Computing it once at build time turns risk analysis into a lookup.
    """
    import networkx as nx

    from core.graph_builder import build_trade_graph
    from core.risk_analyzer import _build_distance_graph

    records: List[pd.DataFrame] = []
    for sector in SUPPORTED_SECTORS:
        sector_edges = edges[edges["sector"] == sector]
        if sector_edges.empty:
            continue

        graph = build_trade_graph(sector_edges, sector=sector)
        pagerank = nx.pagerank(graph, weight="weight")
        betweenness = nx.betweenness_centrality(
            _build_distance_graph(graph), weight="weight", normalized=True
        )

        frame = pd.DataFrame(
            {
                "country": list(graph.nodes),
                "pagerank_centrality": [float(pagerank.get(node, 0.0)) for node in graph.nodes],
                "betweenness_centrality": [float(betweenness.get(node, 0.0)) for node in graph.nodes],
            }
        )
        frame["year"] = int(sector_edges["year"].iloc[0])
        frame["sector"] = sector
        records.append(frame)

    if not records:
        return pd.DataFrame(
            columns=["year", "sector", "country", "pagerank_centrality", "betweenness_centrality"]
        )

    return pd.concat(records, ignore_index=True).loc[
        :, ["year", "sector", "country", "pagerank_centrality", "betweenness_centrality"]
    ]


def summarize_country_year(edges: pd.DataFrame) -> pd.DataFrame:
    """Collapse aggregated edges into per-country yearly totals."""
    exports = (
        edges.groupby(["exporter", "year", "sector"], as_index=False)
        .agg(exports=("trade_value", "sum"), export_partners=("importer", "nunique"))
        .rename(columns={"exporter": "country"})
    )
    imports = (
        edges.groupby(["importer", "year", "sector"], as_index=False)
        .agg(imports=("trade_value", "sum"), import_partners=("exporter", "nunique"))
        .rename(columns={"importer": "country"})
    )

    totals = exports.merge(imports, on=["country", "year", "sector"], how="outer")
    numeric = ["exports", "imports", "export_partners", "import_partners"]
    totals[numeric] = totals[numeric].fillna(0.0)

    totals["trade_total"] = totals["imports"] + totals["exports"]
    safe_total = totals["trade_total"].replace(0.0, pd.NA)
    totals["import_dependency_ratio"] = (totals["imports"] / safe_total).fillna(0.0)
    totals["export_dependency_ratio"] = (totals["exports"] / safe_total).fillna(0.0)
    totals["export_partners"] = totals["export_partners"].astype(int)
    totals["import_partners"] = totals["import_partners"].astype(int)

    return totals.sort_values(["sector", "country", "year"]).reset_index(drop=True)


def build(
    dataset_dir: Path,
    cache_dir: Path,
    years: Optional[Iterable[int]],
    force: bool,
    with_centrality: bool = True,
) -> None:
    """Build the parquet cache for the requested years."""
    if not is_baci_directory(dataset_dir):
        raise SystemExit(f"'{dataset_dir}' does not contain BACI yearly files.")

    all_years = available_years(dataset_dir)
    selected = sorted(set(years) & set(all_years)) if years else all_years
    if not selected:
        raise SystemExit("No matching BACI years were found for the requested selection.")

    edges_dir = cache_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    country_codes = _load_country_codes_cached(str(dataset_dir))
    country_map = country_codes.set_index("country_code")["country_name"].to_dict()

    # Bundle the code table so the cache works without the raw dataset present.
    country_codes.to_parquet(countries_path(cache_dir), index=False, compression="snappy")

    quality_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    centrality_frames: List[pd.DataFrame] = []
    started = time.time()

    for position, year in enumerate(selected, start=1):
        target = edges_path(cache_dir, year)
        if target.exists() and not force:
            print(f"[{position}/{len(selected)}] {year}  cached, skipping", flush=True)
            existing = pd.read_parquet(target)
            summary_frames.append(summarize_country_year(existing))
            if with_centrality:
                centrality_frames.append(compute_centrality(existing))
            continue

        year_started = time.time()
        edges, quality_rows = build_year(dataset_dir, year, country_map)
        edges.to_parquet(target, index=False, compression="snappy")

        quality_frames.append(pd.DataFrame(quality_rows))
        summary_frames.append(summarize_country_year(edges))

        centrality_seconds = 0.0
        if with_centrality:
            centrality_started = time.time()
            centrality_frames.append(compute_centrality(edges))
            centrality_seconds = time.time() - centrality_started

        size_mb = target.stat().st_size / 1_048_576
        print(
            f"[{position}/{len(selected)}] {year}  "
            f"{len(edges):>7,} edges  {size_mb:5.1f} MB  "
            f"{time.time() - year_started:5.1f}s (centrality {centrality_seconds:4.1f}s)",
            flush=True,
        )

    # Rebuilding a subset must not discard the rows already on disk for other
    # years, so merge with whatever the previous build left behind.
    country_year = pd.concat(summary_frames, ignore_index=True)
    quality = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()

    if quality_path(cache_dir).exists():
        previous = pd.read_parquet(quality_path(cache_dir))
        if not quality.empty:
            previous = previous[~previous["year"].isin(quality["year"].unique())]
        quality = pd.concat([previous, quality], ignore_index=True)

    country_year = country_year.sort_values(["sector", "country", "year"]).reset_index(drop=True)
    country_year.to_parquet(country_year_path(cache_dir), index=False, compression="snappy")

    if not quality.empty:
        quality = quality.sort_values(["year", "sector"]).reset_index(drop=True)
        quality.to_parquet(quality_path(cache_dir), index=False, compression="snappy")

    if centrality_frames:
        centrality = pd.concat(centrality_frames, ignore_index=True)
        centrality = centrality.sort_values(["year", "sector", "country"]).reset_index(drop=True)
        centrality.to_parquet(centrality_path(cache_dir), index=False, compression="snappy")

    built_years = sorted(
        int(path.stem.rsplit("_", 1)[1]) for path in edges_dir.glob("edges_*.parquet")
    )
    meta_path(cache_dir).write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_directory": str(dataset_dir.resolve()),
                "years": built_years,
                "sectors": list(SUPPORTED_SECTORS),
                "countries": int(country_year["country"].nunique()),
                "has_centrality": centrality_path(cache_dir).exists(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_mb = sum(path.stat().st_size for path in cache_dir.rglob("*.parquet")) / 1_048_576
    print(
        f"\nCache ready in {time.time() - started:.1f}s  |  "
        f"{len(built_years)} years  |  {country_year['country'].nunique()} countries  |  {total_mb:.1f} MB",
        flush=True,
    )
    print(f"Location: {cache_dir.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BACI parquet cache.")
    parser.add_argument("--dataset", default="dataset", help="Directory holding the BACI yearly CSVs.")
    parser.add_argument("--cache", default="cache", help="Directory to write the parquet cache into.")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Limit the build to these years.")
    parser.add_argument("--force", action="store_true", help="Rebuild years that are already cached.")
    parser.add_argument(
        "--skip-centrality",
        action="store_true",
        help="Skip precomputing PageRank and betweenness. Builds faster; risk queries stay slow.",
    )
    args = parser.parse_args()

    build(
        Path(args.dataset),
        Path(args.cache),
        args.years,
        args.force,
        with_centrality=not args.skip_centrality,
    )


if __name__ == "__main__":
    main()
