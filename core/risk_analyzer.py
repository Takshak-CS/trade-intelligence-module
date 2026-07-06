"""Risk analysis for trade graphs."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd


def analyze_trade_risk(
    graph: nx.DiGraph,
    pagerank_weight: float = 0.4,
    betweenness_weight: float = 0.3,
    concentration_weight: float = 0.3,
    top_n_partners: int = 3,
) -> pd.DataFrame:
    """Compute centrality, concentration, and a composite risk score."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot analyze risk on an empty trade graph.")

    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(_build_distance_graph(graph), weight="weight", normalized=True)
    concentration = compute_trade_concentration(graph, top_n=top_n_partners)

    risk_frame = concentration.copy()
    risk_frame["pagerank_centrality"] = risk_frame["country"].map(pagerank).fillna(0.0)
    risk_frame["betweenness_centrality"] = risk_frame["country"].map(betweenness).fillna(0.0)
    risk_frame["pagerank_norm"] = _min_max_normalize(risk_frame["pagerank_centrality"])
    risk_frame["betweenness_norm"] = _min_max_normalize(risk_frame["betweenness_centrality"])
    risk_frame["risk_score"] = (
        pagerank_weight * risk_frame["pagerank_norm"]
        + betweenness_weight * risk_frame["betweenness_norm"]
        + concentration_weight * risk_frame["trade_concentration"]
    )

    return risk_frame.sort_values("risk_score", ascending=False).reset_index(drop=True)


def compute_trade_concentration(graph: nx.DiGraph, top_n: int = 3) -> pd.DataFrame:
    """Measure how concentrated each country's trade is among its largest partners."""
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
                "top_partners": [partner for partner, _ in sorted_partners[:top_n]],
                "partner_count": int(len(sorted_partners)),
            }
        )

    return pd.DataFrame(records)


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
