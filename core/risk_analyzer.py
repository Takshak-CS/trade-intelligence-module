"""Risk analysis for trade graphs."""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import pandas as pd


def analyze_trade_risk(
    graph: nx.DiGraph,
    pagerank_weight: float = 0.4,
    betweenness_weight: float = 0.3,
    concentration_weight: float = 0.3,
    top_n_partners: int = 3,
    centrality: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute centrality, concentration, and a composite risk score.

    Weighted betweenness dominates the cost of this function, so callers that
    already hold a precomputed centrality table (see ``core.baci_cache``) can
    pass it in to skip the graph traversal entirely.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot analyze risk on an empty trade graph.")

    if centrality is None:
        pagerank = nx.pagerank(graph, weight="weight")
        betweenness = nx.betweenness_centrality(
            _build_distance_graph(graph), weight="weight", normalized=True
        )
    else:
        pagerank = dict(zip(centrality["country"], centrality["pagerank_centrality"]))
        betweenness = dict(zip(centrality["country"], centrality["betweenness_centrality"]))

    concentration = compute_trade_concentration(graph, top_n=top_n_partners)

    risk_frame = concentration.copy()
    risk_frame["pagerank_centrality"] = risk_frame["country"].map(pagerank).fillna(0.0)
    risk_frame["betweenness_centrality"] = risk_frame["country"].map(betweenness).fillna(0.0)
    risk_frame["pagerank_norm"] = _min_max_normalize(risk_frame["pagerank_centrality"])
    risk_frame["betweenness_norm"] = _min_max_normalize(risk_frame["betweenness_centrality"])
    risk_frame["risk_score"] = (
        pagerank_weight * risk_frame["pagerank_norm"]
        + betweenness_weight * risk_frame["betweenness_norm"]
        + concentration_weight * risk_frame["concentration_index"]
    )

    return risk_frame.sort_values("risk_score", ascending=False).reset_index(drop=True)


def compute_trade_concentration(graph: nx.DiGraph, top_n: int = 3) -> pd.DataFrame:
    """Measure how concentrated each country's trade is among its partners.

    Two measures are reported. ``trade_concentration`` is the share held by the
    top N partners, which is easy to read but saturates at 1.0 for any country
    with N or fewer partners — misleading on sparse sector subgraphs where a
    small country legitimately has two or three partners.

    ``concentration_index`` is the Herfindahl-Hirschman Index over the full
    partner distribution. It degrades gracefully at any partner count, is the
    standard measure of market concentration, and is what feeds the risk score.
    """
    records = []
    for country in sorted(graph.nodes):
        partner_totals = _country_partner_totals(graph, country)
        sorted_partners = sorted(partner_totals.items(), key=lambda item: item[1], reverse=True)
        total_trade = float(sum(partner_totals.values()))
        top_trade = float(sum(value for _, value in sorted_partners[:top_n]))
        concentration = top_trade / total_trade if total_trade else 0.0

        records.append(
            {
                "country": country,
                "trade_concentration": float(concentration),
                "concentration_index": _herfindahl_index(partner_totals),
                "top_partners": [partner for partner, _ in sorted_partners[:top_n]],
                "partner_count": int(len(sorted_partners)),
            }
        )

    return pd.DataFrame(records)


def _herfindahl_index(partner_totals: Dict[str, float]) -> float:
    """Herfindahl-Hirschman Index of a country's partner distribution.

    Returns the sum of squared partner shares: 1.0 when a single partner takes
    all the trade, approaching 0 as trade spreads evenly across many partners.
    """
    total = float(sum(partner_totals.values()))
    if total <= 0:
        return 0.0
    return float(sum((value / total) ** 2 for value in partner_totals.values()))


def _country_partner_totals(graph: nx.DiGraph, country: str) -> Dict[str, float]:
    """Aggregate bilateral trade totals between a country and each partner."""
    partner_totals: Dict[str, float] = {}

    for _, importer, edge_data in graph.out_edges(country, data=True):
        partner_totals[importer] = partner_totals.get(importer, 0.0) + float(edge_data.get("weight", 0.0))

    for exporter, _, edge_data in graph.in_edges(country, data=True):
        partner_totals[exporter] = partner_totals.get(exporter, 0.0) + float(edge_data.get("weight", 0.0))

    return partner_totals


def _build_distance_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Convert trade weights into distance weights for shortest-path metrics."""
    distance_graph = nx.DiGraph()
    distance_graph.add_nodes_from(graph.nodes)

    for exporter, importer, edge_data in graph.edges(data=True):
        trade_value = float(edge_data.get("weight", 0.0))
        if trade_value <= 0:
            continue
        distance_graph.add_edge(exporter, importer, weight=1.0 / trade_value)

    return distance_graph


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a numeric series to the [0, 1] interval."""
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum - minimum <= 0:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)
