"""Trade bloc detection and evolution.

Countries do not trade uniformly with everyone. They cluster: dense internal
trade, thinner links outward. Those clusters are trade blocs, and they are not
always the ones on paper — the blocs that emerge from actual flows can cut
across formal agreements.

Detection runs Louvain community detection over the undirected projection of the
trade graph, where each edge carries the full two-way value of the relationship.
The result is deterministic for a given graph because the seed is fixed, which
matters: a bloc assignment that changed between two identical queries would be
impossible to defend.

This module is also the intended join point with the policy stance module. Blocs
found here describe who trades together; blocs found from UN voting similarity
describe who votes together. Where the two disagree is the interesting part.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import networkx as nx
import pandas as pd

LOUVAIN_SEED = 42
DEFAULT_RESOLUTION = 1.0


def undirected_projection(graph: nx.DiGraph) -> nx.Graph:
    """Collapse a directed trade graph into an undirected relationship graph.

    Each undirected edge carries the sum of trade in both directions, which is
    the right weight for community detection: bloc membership is about the
    strength of a relationship, not its direction.
    """
    projected = nx.Graph()
    projected.add_nodes_from(graph.nodes)

    for exporter, importer, edge_data in graph.edges(data=True):
        weight = float(edge_data.get("weight", 0.0))
        if weight <= 0:
            continue
        if projected.has_edge(exporter, importer):
            projected[exporter][importer]["weight"] += weight
        else:
            projected.add_edge(exporter, importer, weight=weight)

    return projected


def detect_trade_blocs(
    graph: nx.DiGraph,
    resolution: float = DEFAULT_RESOLUTION,
) -> pd.DataFrame:
    """Assign every country to a trade bloc and measure how inward-facing it is.

    ``internal_trade_share`` is the fraction of a country's trade that stays
    inside its own bloc. A high value means the country's economic life is
    largely contained within its cluster; a low value means it sits on a
    boundary and trades outward as much as inward.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot detect trade blocs on an empty trade graph.")

    projected = undirected_projection(graph)
    communities = nx.community.louvain_communities(
        projected,
        weight="weight",
        resolution=float(resolution),
        seed=LOUVAIN_SEED,
    )

    # Order blocs by economic weight so bloc 0 is always the largest. Without
    # this the integer labels would be arbitrary and unstable across years.
    sized = sorted(
        communities,
        key=lambda members: sum(
            projected[node][neighbour]["weight"]
            for node in members
            for neighbour in projected.neighbors(node)
        ),
        reverse=True,
    )

    assignment: Dict[str, int] = {}
    for bloc_id, members in enumerate(sized):
        for member in members:
            assignment[member] = bloc_id

    # Name each bloc after its largest trading member: "the China bloc" reads
    # better than "bloc 0" and stays meaningful in a summary sentence.
    node_strength = {
        node: sum(projected[node][neighbour]["weight"] for neighbour in projected.neighbors(node))
        for node in projected.nodes
    }
    bloc_names = {
        bloc_id: max(members, key=lambda member: node_strength.get(member, 0.0))
        for bloc_id, members in enumerate(sized)
    }

    records = []
    for country in sorted(graph.nodes):
        bloc_id = assignment.get(country, -1)
        total = node_strength.get(country, 0.0)
        internal = sum(
            projected[country][neighbour]["weight"]
            for neighbour in projected.neighbors(country)
            if assignment.get(neighbour, -2) == bloc_id
        )
        records.append(
            {
                "country": country,
                "bloc_id": int(bloc_id),
                "bloc_name": bloc_names.get(bloc_id, ""),
                "bloc_size": int(len(sized[bloc_id])) if 0 <= bloc_id < len(sized) else 0,
                "total_trade": float(total),
                "internal_trade": float(internal),
                "internal_trade_share": float(internal / total) if total > 0 else 0.0,
            }
        )

    frame = pd.DataFrame(records)
    frame.attrs["modularity"] = float(
        nx.community.modularity(projected, sized, weight="weight", resolution=float(resolution))
    )
    frame.attrs["bloc_count"] = int(len(sized))
    return frame


def summarize_blocs(assignment: pd.DataFrame, top_members: int = 5) -> pd.DataFrame:
    """Roll a country-level bloc assignment up into one row per bloc."""
    if assignment.empty:
        return pd.DataFrame(columns=["bloc_id", "bloc_name", "member_count", "members", "total_trade"])

    records = []
    for bloc_id, group in assignment.groupby("bloc_id"):
        ordered = group.sort_values("total_trade", ascending=False)
        records.append(
            {
                "bloc_id": int(bloc_id),
                "bloc_name": str(ordered.iloc[0]["bloc_name"]),
                "member_count": int(len(group)),
                "members": ordered["country"].head(top_members).tolist(),
                "all_members": ordered["country"].tolist(),
                "total_trade": float(group["total_trade"].sum()),
                "mean_internal_share": float(group["internal_trade_share"].mean()),
            }
        )

    return pd.DataFrame(records).sort_values("total_trade", ascending=False).reset_index(drop=True)


def inter_bloc_flows(graph: nx.DiGraph, assignment: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade flowing between blocs, in both directions."""
    bloc_of = dict(zip(assignment["country"], assignment["bloc_id"]))
    name_of = dict(zip(assignment["bloc_id"], assignment["bloc_name"]))

    flows: Dict[tuple, float] = {}
    for exporter, importer, edge_data in graph.edges(data=True):
        source = bloc_of.get(exporter)
        target = bloc_of.get(importer)
        if source is None or target is None:
            continue
        key = (int(source), int(target))
        flows[key] = flows.get(key, 0.0) + float(edge_data.get("weight", 0.0))

    records = [
        {
            "from_bloc": source,
            "from_bloc_name": name_of.get(source, ""),
            "to_bloc": target,
            "to_bloc_name": name_of.get(target, ""),
            "trade_value": float(value),
            "is_internal": source == target,
        }
        for (source, target), value in flows.items()
    ]

    if not records:
        return pd.DataFrame(
            columns=["from_bloc", "from_bloc_name", "to_bloc", "to_bloc_name", "trade_value", "is_internal"]
        )

    return pd.DataFrame(records).sort_values("trade_value", ascending=False).reset_index(drop=True)


def track_bloc_evolution(
    assignments_by_year: Dict[int, pd.DataFrame],
    country: Optional[str] = None,
) -> pd.DataFrame:
    """Follow bloc membership across years.

    Louvain labels are not comparable across independent runs, so this tracks
    membership by the bloc's named anchor country rather than by integer id.
    ``switched`` flags the years where a country changed bloc, which is the
    signal worth looking at — a country moving between trading clusters is a
    slow geopolitical realignment showing up in the data.
    """
    records: List[dict] = []
    previous: Dict[str, str] = {}

    for year in sorted(assignments_by_year):
        assignment = assignments_by_year[year]
        selected = assignment if country is None else assignment[assignment["country"] == country]

        for row in selected.itertuples(index=False):
            prior = previous.get(row.country)
            records.append(
                {
                    "year": int(year),
                    "country": row.country,
                    "bloc_name": row.bloc_name,
                    "bloc_size": int(row.bloc_size),
                    "internal_trade_share": float(row.internal_trade_share),
                    "previous_bloc": prior or "",
                    "switched": bool(prior is not None and prior != row.bloc_name),
                }
            )

        for row in assignment.itertuples(index=False):
            previous[row.country] = row.bloc_name

    if not records:
        return pd.DataFrame(
            columns=["year", "country", "bloc_name", "bloc_size", "internal_trade_share", "previous_bloc", "switched"]
        )

    return pd.DataFrame(records).sort_values(["country", "year"]).reset_index(drop=True)
