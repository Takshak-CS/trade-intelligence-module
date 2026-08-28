"""Sector fragility analysis.

The rest of the module can filter to a sector, but it never compares sectors to
each other — so it cannot answer the question that actually matters for supply
chain risk: *where* is a country brittle?

A country is fragile in a sector when two things hold at once. It leans on
imports for that sector, and those imports arrive from very few suppliers.
Either condition alone is survivable. Together they are a chokepoint: a single
supplier disruption has nowhere to route around.

This module scores that per country and sector, and separately ranks sectors by
how concentrated global supply is, which tells you how hard substitution would
be if a chokepoint ever closed.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import networkx as nx
import pandas as pd

# How much of the fragility score comes from supplier concentration versus how
# much the country leans on the sector. Concentration is weighted higher: three
# suppliers for a small import need is more dangerous than thirty suppliers for
# a large one.
CONCENTRATION_WEIGHT = 0.6
RELIANCE_WEIGHT = 0.4


def _herfindahl(values: Mapping[str, float]) -> float:
    """Herfindahl-Hirschman Index over a distribution of trade values."""
    total = float(sum(values.values()))
    if total <= 0:
        return 0.0
    return float(sum((value / total) ** 2 for value in values.values()))


def _supplier_shares(graph: nx.DiGraph, country: str) -> Dict[str, float]:
    """Value imported by a country from each of its suppliers."""
    return {
        exporter: float(edge_data.get("weight", 0.0))
        for exporter, _, edge_data in graph.in_edges(country, data=True)
    }


def _buyer_shares(graph: nx.DiGraph, country: str) -> Dict[str, float]:
    """Value exported by a country to each of its buyers."""
    return {
        importer: float(edge_data.get("weight", 0.0))
        for _, importer, edge_data in graph.out_edges(country, data=True)
    }


def compute_sector_fragility(
    graphs_by_sector: Mapping[str, nx.DiGraph],
    country: Optional[str] = None,
    min_sector_imports: float = 0.0,
) -> pd.DataFrame:
    """Score every country's fragility in each sector.

    Pass ``country`` to get one country's profile across all sectors, which is
    the shape you want for "where is India brittle"; omit it to score everyone,
    which is the shape you want for "who is most brittle in energy".

    An unfiltered global ranking is dominated by micro-territories: a dependency
    that is real but uninteresting, because a handful of islands importing from
    their nearest neighbour will always score near 1.0. ``min_sector_imports``
    sets a floor on sector import value so the ranking surfaces economies whose
    fragility carries systemic weight. It is ignored when scoring a single named
    country, since there the caller has already chosen the subject.
    """
    specific_sectors = {
        sector: graph for sector, graph in graphs_by_sector.items() if sector != "all"
    }
    if not specific_sectors:
        raise ValueError("Sector fragility needs at least one sector graph besides 'all'.")

    # A country's reliance on a sector is measured against its imports across
    # the sectors we can actually see, not against its whole economy.
    total_imports: Dict[str, float] = {}
    for graph in specific_sectors.values():
        for node in graph.nodes:
            total_imports[node] = total_imports.get(node, 0.0) + float(
                sum(data.get("weight", 0.0) for _, _, data in graph.in_edges(node, data=True))
            )

    records = []
    for sector, graph in specific_sectors.items():
        # How hard would it be to find another source anywhere in the world?
        global_exporters = {
            node for node in graph.nodes if graph.out_degree(node) > 0
        }

        candidates = graph.nodes if country is None else [country]
        for node in candidates:
            if node not in graph:
                continue

            suppliers = _supplier_shares(graph, node)
            buyers = _buyer_shares(graph, node)
            sector_imports = float(sum(suppliers.values()))
            sector_exports = float(sum(buyers.values()))

            supplier_concentration = _herfindahl(suppliers)
            country_total = total_imports.get(node, 0.0)
            sector_reliance = sector_imports / country_total if country_total > 0 else 0.0

            top_supplier, top_supplier_value = ("", 0.0)
            if suppliers:
                top_supplier, top_supplier_value = max(suppliers.items(), key=lambda item: item[1])

            records.append(
                {
                    "country": node,
                    "sector": sector,
                    "sector_imports": sector_imports,
                    "sector_exports": sector_exports,
                    "supplier_count": int(len(suppliers)),
                    "buyer_count": int(len(buyers)),
                    "supplier_concentration": float(supplier_concentration),
                    "sector_reliance": float(sector_reliance),
                    "top_supplier": top_supplier,
                    "top_supplier_share": float(
                        top_supplier_value / sector_imports if sector_imports > 0 else 0.0
                    ),
                    "global_supplier_pool": int(len(global_exporters)),
                    "fragility_score": float(
                        CONCENTRATION_WEIGHT * supplier_concentration
                        + RELIANCE_WEIGHT * sector_reliance
                    ),
                }
            )

    if not records:
        raise ValueError("No sector fragility could be computed for the requested scope.")

    frame = pd.DataFrame(records)
    if country is None and min_sector_imports > 0:
        filtered = frame[frame["sector_imports"] >= float(min_sector_imports)]
        # Never hand back an empty ranking because the threshold was set too
        # high; fall back to the unfiltered frame instead.
        if not filtered.empty:
            frame = filtered

    return frame.sort_values("fragility_score", ascending=False).reset_index(drop=True)


def rank_sector_fragility(graphs_by_sector: Mapping[str, nx.DiGraph]) -> pd.DataFrame:
    """Rank sectors by how concentrated global supply is.

    This is the network-level view: a sector where a handful of countries supply
    the world is one where any single disruption propagates widely, regardless
    of which importer you happen to be looking at.
    """
    specific_sectors = {
        sector: graph for sector, graph in graphs_by_sector.items() if sector != "all"
    }
    if not specific_sectors:
        raise ValueError("Sector ranking needs at least one sector graph besides 'all'.")

    records = []
    for sector, graph in specific_sectors.items():
        exports_by_country = {
            node: float(sum(data.get("weight", 0.0) for _, _, data in graph.out_edges(node, data=True)))
            for node in graph.nodes
        }
        active = {node: value for node, value in exports_by_country.items() if value > 0}
        total = float(sum(active.values()))

        ranked = sorted(active.items(), key=lambda item: item[1], reverse=True)
        top_three_share = float(sum(value for _, value in ranked[:3]) / total) if total > 0 else 0.0

        records.append(
            {
                "sector": sector,
                "exporter_count": int(len(active)),
                "supply_concentration": _herfindahl(active),
                "top_three_share": top_three_share,
                "dominant_supplier": ranked[0][0] if ranked else "",
                "dominant_supplier_share": float(ranked[0][1] / total) if ranked and total > 0 else 0.0,
                "total_trade": total,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("supply_concentration", ascending=False)
        .reset_index(drop=True)
    )


def fragility_profile(
    graphs_by_sector: Mapping[str, nx.DiGraph],
    country: str,
) -> dict:
    """Describe where a single country is most and least exposed."""
    fragility = compute_sector_fragility(graphs_by_sector, country=country)
    if fragility.empty:
        raise ValueError(f"No sector fragility could be computed for '{country}'.")

    most = fragility.iloc[0]
    least = fragility.iloc[-1]

    return {
        "country": country,
        "sectors": fragility.to_dict(orient="records"),
        "most_fragile_sector": str(most["sector"]),
        "most_fragile_score": float(most["fragility_score"]),
        "least_fragile_sector": str(least["sector"]),
        "least_fragile_score": float(least["fragility_score"]),
    }
