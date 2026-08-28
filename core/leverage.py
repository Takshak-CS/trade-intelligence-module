"""Asymmetric trade leverage.

Volume alone does not tell you who holds power in a trade relationship. Two
countries can trade the same amount with each other and be in completely
different positions: what matters is what share of each side's total trade that
flow represents.

If 40% of Djibouti's trade is with China but only 0.1% of China's trade is with
Djibouti, the relationship is worth the same dollars to both and means something
entirely different to each. That gap is the leverage, and it is what makes trade
usable as an instrument of policy.

Two views are provided:

- ``compute_leverage_pairs`` measures the asymmetry in each bilateral
  relationship and names which side holds the advantage.
- ``compute_country_leverage`` aggregates those pairs into a per-country
  position: how much leverage a country holds over others, how much others hold
  over it, and which single partner it is most exposed to.
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import pandas as pd

# Pairs below this share of either side's trade are noise for ranking purposes:
# a relationship worth 0.001% to both countries has a meaningless ratio.
DEFAULT_MIN_SHARE = 0.005


def bilateral_totals(graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Total trade flowing in both directions for every country pair.

    Pairs are keyed in sorted order so that (A, B) and (B, A) collapse into one
    entry carrying the full two-way relationship.
    """
    totals: Dict[Tuple[str, str], float] = {}
    for exporter, importer, edge_data in graph.edges(data=True):
        key = (exporter, importer) if exporter <= importer else (importer, exporter)
        totals[key] = totals.get(key, 0.0) + float(edge_data.get("weight", 0.0))
    return totals


def country_totals(graph: nx.DiGraph) -> Dict[str, float]:
    """Total trade activity per country, counting imports and exports."""
    totals: Dict[str, float] = {node: 0.0 for node in graph.nodes}
    for exporter, importer, edge_data in graph.edges(data=True):
        weight = float(edge_data.get("weight", 0.0))
        totals[exporter] = totals.get(exporter, 0.0) + weight
        totals[importer] = totals.get(importer, 0.0) + weight
    return totals


def compute_leverage_pairs(
    graph: nx.DiGraph,
    min_share: float = DEFAULT_MIN_SHARE,
    min_bilateral_trade: float = 0.0,
) -> pd.DataFrame:
    """Measure dependence asymmetry for every bilateral trade relationship.

    For each pair, dependence is the share of a country's total trade that runs
    through this one partner. ``asymmetry`` is the gap between the two sides:
    positive values mean ``leverage_holder`` is the less dependent party and
    therefore the one that could absorb a rupture more easily.

    ``min_bilateral_trade`` filters out relationships too small to matter in
    absolute terms. Without it the ranking fills up with micro-territories
    whose dependence is genuine but carries no systemic weight.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot compute leverage on an empty trade graph.")

    totals = country_totals(graph)
    pairs = bilateral_totals(graph)

    records = []
    for (first, second), bilateral in pairs.items():
        first_total = totals.get(first, 0.0)
        second_total = totals.get(second, 0.0)
        if first_total <= 0 or second_total <= 0:
            continue

        if bilateral < min_bilateral_trade:
            continue

        first_dependence = bilateral / first_total
        second_dependence = bilateral / second_total
        if max(first_dependence, second_dependence) < min_share:
            continue

        # The country with the LOWER dependence holds the leverage: it gives up
        # a smaller share of its trade if the relationship breaks.
        if first_dependence <= second_dependence:
            holder, exposed = first, second
            holder_dependence, exposed_dependence = first_dependence, second_dependence
        else:
            holder, exposed = second, first
            holder_dependence, exposed_dependence = second_dependence, first_dependence

        records.append(
            {
                "leverage_holder": holder,
                "exposed_country": exposed,
                "bilateral_trade": float(bilateral),
                "holder_dependence": float(holder_dependence),
                "exposed_dependence": float(exposed_dependence),
                "asymmetry": float(exposed_dependence - holder_dependence),
                "exposure_ratio": float(
                    exposed_dependence / holder_dependence if holder_dependence > 0 else float("inf")
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "leverage_holder",
                "exposed_country",
                "bilateral_trade",
                "holder_dependence",
                "exposed_dependence",
                "asymmetry",
                "exposure_ratio",
            ]
        )

    frame = pd.DataFrame(records)
    return frame.sort_values("asymmetry", ascending=False).reset_index(drop=True)


def compute_country_leverage(
    graph: nx.DiGraph,
    min_share: float = DEFAULT_MIN_SHARE,
    min_bilateral_trade: float = 0.0,
) -> pd.DataFrame:
    """Aggregate pairwise asymmetry into a per-country leverage position.

    ``leverage_held`` sums the asymmetry a country enjoys across all the
    relationships where it is the stronger side; ``leverage_exposed`` sums the
    asymmetry working against it. The difference is its net position in the
    network.
    """
    pairs = compute_leverage_pairs(
        graph, min_share=min_share, min_bilateral_trade=min_bilateral_trade
    )
    countries = sorted(graph.nodes)

    held: Dict[str, float] = {country: 0.0 for country in countries}
    exposed: Dict[str, float] = {country: 0.0 for country in countries}
    holds_over: Dict[str, int] = {country: 0 for country in countries}
    exposed_to: Dict[str, int] = {country: 0 for country in countries}

    for row in pairs.itertuples(index=False):
        held[row.leverage_holder] = held.get(row.leverage_holder, 0.0) + row.asymmetry
        exposed[row.exposed_country] = exposed.get(row.exposed_country, 0.0) + row.asymmetry
        holds_over[row.leverage_holder] = holds_over.get(row.leverage_holder, 0) + 1
        exposed_to[row.exposed_country] = exposed_to.get(row.exposed_country, 0) + 1

    # The single relationship a country can least afford to lose.
    totals = country_totals(graph)
    bilateral = bilateral_totals(graph)
    top_partner: Dict[str, str] = {}
    top_dependence: Dict[str, float] = {country: 0.0 for country in countries}

    for (first, second), value in bilateral.items():
        for country, partner in ((first, second), (second, first)):
            country_total = totals.get(country, 0.0)
            if country_total <= 0:
                continue
            dependence = value / country_total
            if dependence > top_dependence.get(country, 0.0):
                top_dependence[country] = dependence
                top_partner[country] = partner

    records = [
        {
            "country": country,
            "leverage_held": float(held.get(country, 0.0)),
            "leverage_exposed": float(exposed.get(country, 0.0)),
            "net_leverage": float(held.get(country, 0.0) - exposed.get(country, 0.0)),
            "partners_dominated": int(holds_over.get(country, 0)),
            "partners_dependent_on": int(exposed_to.get(country, 0)),
            "most_critical_partner": top_partner.get(country, ""),
            "critical_partner_dependence": float(top_dependence.get(country, 0.0)),
        }
        for country in countries
    ]

    return (
        pd.DataFrame(records)
        .sort_values("net_leverage", ascending=False)
        .reset_index(drop=True)
    )


def leverage_profile(graph: nx.DiGraph, country: str, limit: int = 5) -> dict:
    """Describe one country's leverage position and its most lopsided relationships."""
    if country not in graph:
        raise ValueError(f"Country '{country}' is not present in the trade graph.")

    pairs = compute_leverage_pairs(graph)
    summary = compute_country_leverage(graph)
    row = summary[summary["country"] == country]
    if row.empty:
        raise ValueError(f"No leverage position could be computed for '{country}'.")

    holds = pairs[pairs["leverage_holder"] == country].head(limit)
    vulnerable = pairs[pairs["exposed_country"] == country].head(limit)

    return {
        "country": country,
        "position": row.iloc[0].to_dict(),
        "holds_leverage_over": holds.to_dict(orient="records"),
        "vulnerable_to": vulnerable.to_dict(orient="records"),
    }
