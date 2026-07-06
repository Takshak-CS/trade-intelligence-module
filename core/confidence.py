"""Confidence scoring utilities for trade insights."""

from __future__ import annotations

from typing import Mapping, Optional

import networkx as nx



def build_confidence_assessment(
    data_quality: Optional[Mapping[str, float]],
    graph: nx.DiGraph,
    country: str,
    propagation_steps: int = 1,
) -> dict:
    """Build a normalized confidence score and explanation from interpretable components."""
    data_score, data_reason = data_completeness_score(data_quality)
    connectivity_score, connectivity_reason = graph_connectivity_score(graph, country)
    propagation_score, propagation_reason = propagation_reliability_score(propagation_steps)

    components = {
        "data_completeness": data_score,
        "graph_connectivity": connectivity_score,
        "propagation_reliability": propagation_score,
    }
    confidence = _clamp(sum(components.values()) / len(components))

    reasons = {
        "data_completeness": data_reason,
        "graph_connectivity": connectivity_reason,
        "propagation_reliability": propagation_reason,
    }
    limiting_component = min(components, key=components.get)

    return {
        "score": confidence,
        "reason": reasons[limiting_component],
        "components": components,
    }


def combine_forecast_confidence_assessment(
    heuristic_assessment: Mapping[str, object],
    model_confidence: float,
    model_weight: float = 0.6,
    heuristic_weight: float = 0.4,
) -> dict:
    """Combine forecast model confidence with the existing heuristic confidence."""
    heuristic_score = _clamp(float(heuristic_assessment.get("score", 0.0)))
    bounded_model_confidence = _clamp(float(model_confidence))
    total_weight = float(model_weight + heuristic_weight)
    if total_weight <= 0:
        raise ValueError("Forecast confidence weights must sum to a positive value.")

    combined_score = _clamp(
        (
            (bounded_model_confidence * model_weight)
            + (heuristic_score * heuristic_weight)
        )
        / total_weight
    )

    heuristic_reason = str(heuristic_assessment.get("reason", "")).strip()
    if bounded_model_confidence <= heuristic_score:
        reason = "Confidence is most sensitive to the forecast model fit over the available history."
    else:
        reason = heuristic_reason or "Confidence is supported by the underlying trade data and network context."

    components = {
        "model_confidence": bounded_model_confidence,
        "heuristic_confidence": heuristic_score,
    }
    heuristic_components = heuristic_assessment.get("components", {})
    if isinstance(heuristic_components, Mapping):
        for component_name, component_score in heuristic_components.items():
            components[f"heuristic_{component_name}"] = _clamp(float(component_score))

    return {
        "score": combined_score,
        "reason": reason,
        "components": components,
    }



def data_completeness_score(data_quality: Optional[Mapping[str, float]]) -> tuple[float, str]:
    """Score completeness using required-field availability and quantity availability."""
    if not data_quality:
        return 1.0, "Confidence is supported by complete trade records in the selected slice."

    required_availability = float(data_quality.get("required_field_availability", 1.0))
    quantity_availability = data_quality.get("quantity_availability")

    components = [required_availability]
    if quantity_availability is not None:
        components.append(float(quantity_availability))

    score = _clamp(sum(components) / len(components))
    if quantity_availability is not None and float(quantity_availability) < required_availability:
        reason = "Confidence is most sensitive to missing quantity observations in the underlying trade data."
    elif required_availability < 1.0:
        reason = "Confidence is most sensitive to incomplete trade records in the selected data slice."
    else:
        reason = "Confidence is supported by complete trade records in the selected slice."

    return score, reason



def graph_connectivity_score(graph: nx.DiGraph, country: str) -> tuple[float, str]:
    """Score connectivity using country degree and overall graph density."""
    if graph.number_of_nodes() <= 1 or country not in graph:
        return 0.0, "Confidence is most sensitive to sparse trade connections for this country."

    degree_map = dict(graph.degree())
    max_degree = max(degree_map.values(), default=0)
    local_connectivity = degree_map.get(country, 0) / max_degree if max_degree > 0 else 0.0
    density = nx.density(graph)
    score = _clamp((local_connectivity + density) / 2.0)

    if local_connectivity <= density:
        reason = "Confidence is most sensitive to sparse trade connections for this country."
    else:
        reason = "Confidence is most sensitive to low overall network density in the selected trade graph."
    return score, reason



def propagation_reliability_score(propagation_steps: int) -> tuple[float, str]:
    """Score reliability inversely to the number of propagation steps used."""
    used_steps = max(1, int(propagation_steps))
    score = _clamp(1.0 / used_steps)
    if used_steps > 1:
        reason = "Confidence is most sensitive to the number of propagation steps required by the simulation."
    else:
        reason = "Confidence is supported by a direct first-order inference path."
    return score, reason



def _clamp(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return float(max(0.0, min(1.0, value)))
