"""Shock simulation utilities for trade graphs."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd

from core.feature_engineering import compute_total_imports



def simulate_trade_shock(
    graph: nx.DiGraph,
    country: str,
    severity: float,
    steps: int = 3,
    propagation_factor: float = 0.7,
) -> pd.DataFrame:
    """Simulate a supply-side export shock and propagate it through the graph."""
    if country not in graph:
        raise ValueError(f"Country '{country}' is not present in the trade graph.")

    clamped_severity = _clamp(severity)
    baseline_graph = graph.copy()
    working_graph = graph.copy()
    baseline_imports = compute_total_imports(baseline_graph)

    applied_reductions: Dict[str, float] = {node: 0.0 for node in graph.nodes}
    impact_scores: Dict[str, float] = {node: 0.0 for node in graph.nodes}
    actual_steps = 0

    _set_outgoing_reduction(working_graph, baseline_graph, country, clamped_severity)
    applied_reductions[country] = clamped_severity
    impact_scores[country] = clamped_severity

    for _ in range(max(1, steps)):
        actual_steps += 1
        current_imports = compute_total_imports(working_graph)
        changed = False

        for node in working_graph.nodes:
            baseline_import = baseline_imports.get(node, 0.0)
            if baseline_import <= 0:
                continue
            import_loss_ratio = max(0.0, (baseline_import - current_imports.get(node, 0.0)) / baseline_import)
            impact_scores[node] = max(impact_scores[node], _clamp(import_loss_ratio))

        for node in working_graph.nodes:
            desired_reduction = _clamp(impact_scores[node] * propagation_factor)
            if desired_reduction > applied_reductions[node] + 1e-12:
                _set_outgoing_reduction(working_graph, baseline_graph, node, desired_reduction)
                applied_reductions[node] = desired_reduction
                changed = True

        if not changed:
            break

    records = []
    for node, score in impact_scores.items():
        records.append(
            {
                "country": node,
                "impact_score": float(score),
                "applied_export_reduction": float(applied_reductions[node]),
            }
        )

    result = pd.DataFrame(records).sort_values("impact_score", ascending=False).reset_index(drop=True)
    result.attrs["propagation_depth_used"] = actual_steps
    result.attrs["requested_steps"] = max(1, int(steps))
    return result



def _set_outgoing_reduction(
    working_graph: nx.DiGraph,
    baseline_graph: nx.DiGraph,
    country: str,
    reduction: float,
) -> None:
    """Set all outgoing edges from a country to a reduced baseline level."""
    reduction = _clamp(reduction)
    for _, importer, edge_data in baseline_graph.out_edges(country, data=True):
        baseline_weight = float(edge_data.get("weight", 0.0))
        if not working_graph.has_edge(country, importer):
            working_graph.add_edge(country, importer, weight=baseline_weight * (1.0 - reduction))
            continue
        working_graph[country][importer]["weight"] = baseline_weight * (1.0 - reduction)



def _clamp(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return float(max(0.0, min(1.0, value)))
