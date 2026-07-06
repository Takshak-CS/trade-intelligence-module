"""Main execution entry point for the Trade Graph Intelligence Agent."""

from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

from core.confidence import build_confidence_assessment
from core.data_loader import (
    is_baci_directory,
    latest_year,
    load_country_time_series,
    load_trade_data,
    resolve_baci_country,
)
from core.feature_engineering import compute_country_features
from core.forecast import forecast_metric_from_series
from core.graph_builder import build_trade_graph
from core.output_formatter import build_insight, format_agent_output
from core.risk_analyzer import analyze_trade_risk
from core.sector_mapper import normalize_sector, sector_summary_label
from core.shock_simulator import simulate_trade_shock


DEFAULT_DATA_PATH = "dataset" if Path("dataset").exists() else "trade_data.csv"
FORECAST_METRICS = {"imports", "exports"}



def execute(query: dict) -> dict:
    """Execute a trade intelligence query and return structured insights."""
    if not isinstance(query, dict):
        raise ValueError("Query must be a dictionary.")

    query_type = str(query.get("query_type", "risk")).strip().lower()
    data_path = str(query.get("data_path", DEFAULT_DATA_PATH))
    limit = max(1, int(query.get("limit", 10)))
    sector = normalize_sector(query.get("sector", "all"))

    if query_type == "forecast":
        insights = _execute_forecast_query(data_path, query, sector=sector)
        return format_agent_output(insights)

    snapshot_year = _resolve_snapshot_year(data_path, query.get("year"))
    snapshot_data = load_trade_data(data_path, year=snapshot_year, sector=sector)
    graph = build_trade_graph(snapshot_data, sector=sector)
    features = compute_country_features(graph)
    risk = analyze_trade_risk(graph, top_n_partners=max(1, int(query.get("top_n_partners", 3))))

    if query_type == "shock":
        shock_query = dict(query)
        if shock_query.get("country") and is_baci_directory(data_path):
            _, resolved_country = resolve_baci_country(data_path, str(shock_query["country"]))
            shock_query["country"] = resolved_country
        insights = _execute_shock_query(
            graph,
            features,
            risk,
            shock_query,
            snapshot_year,
            limit,
            sector=sector,
            data_quality=snapshot_data.attrs.get("data_quality"),
        )
    elif query_type == "risk":
        insights = _execute_risk_query(
            graph,
            features,
            risk,
            snapshot_year,
            limit,
            sector=sector,
            data_quality=snapshot_data.attrs.get("data_quality"),
        )
    else:
        supported = "risk, shock, forecast"
        raise ValueError(f"Unsupported query_type '{query_type}'. Supported values: {supported}")

    return format_agent_output(insights)



def _execute_risk_query(
    graph: nx.DiGraph,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    snapshot_year: int,
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> List[dict]:
    """Build ranked risk insights for a single graph snapshot."""
    merged = risk.merge(features, on="country", how="left")
    sector_label = sector_summary_label(sector)

    insights: List[dict] = []
    for row in merged.head(limit).itertuples(index=False):
        confidence = build_confidence_assessment(
            data_quality=data_quality,
            graph=graph,
            country=row.country,
            propagation_steps=1,
        )
        summary = (
            f"{sector_label} in year {snapshot_year}: risk score {row.risk_score:.3f}, concentration "
            f"{row.trade_concentration:.2%}, with key partners {', '.join(row.top_partners) if row.top_partners else 'no major partners'}. "
            f"{confidence['reason']}"
        )
        insight = build_insight(
            country=row.country,
            score=float(row.risk_score),
            summary=summary,
            confidence=confidence["score"],
        )
        insight["sector"] = sector
        insight["confidence_reason"] = confidence["reason"]
        insight["confidence_components"] = confidence["components"]
        insights.append(insight)

    return insights



def _execute_shock_query(
    graph: nx.DiGraph,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    query: dict,
    snapshot_year: int,
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> List[dict]:
    """Run a shock simulation and convert it into ranked insights."""
    country = query.get("country")
    if not country:
        raise ValueError("Shock queries must include a 'country'.")

    severity = float(query.get("severity", 0.5))
    steps = max(1, int(query.get("steps", 3)))
    propagation_factor = float(query.get("propagation_factor", 0.7))
    sector_label = sector_summary_label(sector)

    impact = simulate_trade_shock(
        graph=graph,
        country=str(country),
        severity=severity,
        steps=steps,
        propagation_factor=propagation_factor,
    )
    propagation_steps = int(impact.attrs.get("propagation_depth_used", steps))

    merged = (
        impact.merge(features, on="country", how="left")
        .merge(risk[["country", "risk_score"]], on="country", how="left")
        .fillna({"risk_score": 0.0})
    )

    insights: List[dict] = []
    for row in merged.head(limit).itertuples(index=False):
        confidence = build_confidence_assessment(
            data_quality=data_quality,
            graph=graph,
            country=row.country,
            propagation_steps=propagation_steps,
        )
        summary = (
            f"{sector_label} shock from {country} in year {snapshot_year} produces an estimated impact score of "
            f"{row.impact_score:.2%} for {row.country} after {propagation_steps} propagation steps. "
            f"{confidence['reason']}"
        )
        insight = build_insight(
            country=row.country,
            score=float(row.impact_score),
            summary=summary,
            confidence=confidence["score"],
        )
        insight["sector"] = sector
        insight["confidence_reason"] = confidence["reason"]
        insight["confidence_components"] = confidence["components"]
        insights.append(insight)

    return insights



def _execute_forecast_query(data_path: str, query: dict, sector: str) -> List[dict]:
    """Run a multi-year forecast query for imports or exports."""
    country = query.get("country")
    if not country:
        raise ValueError("Forecast queries must include a 'country'.")

    metric = str(query.get("metric", "exports")).strip().lower()
    if metric not in FORECAST_METRICS:
        raise ValueError("Forecast metric must be either 'imports' or 'exports'.")
    forecast_model = str(query.get("forecast_model", query.get("model", "auto"))).strip().lower()

    periods = max(1, int(query.get("periods", 3)))
    time_series_payload = load_country_time_series(data_path, str(country), sector=sector)
    resolved_country = str(time_series_payload["country"])
    forecast_result = forecast_metric_from_series(
        time_series=time_series_payload["time_series"],
        country=resolved_country,
        metric=metric,
        periods=periods,
        model=forecast_model,
    )

    latest_snapshot_year = latest_year(data_path)
    latest_snapshot_data = load_trade_data(data_path, year=latest_snapshot_year, sector=sector)
    latest_graph = build_trade_graph(latest_snapshot_data, sector=sector)
    confidence = build_confidence_assessment(
        data_quality=time_series_payload.get("data_quality"),
        graph=latest_graph,
        country=resolved_country,
        propagation_steps=1,
    )

    forecast_values = [
        {
            "year": int(item["year"]),
            "predicted_value": float(item["predicted_value"]),
        }
        for item in forecast_result["forecast"]
    ]
    first_year = forecast_values[0]["year"]
    final_year = forecast_values[-1]["year"]
    history = forecast_result["history"]
    history_values = [
        {
            "year": int(item["year"]),
            "value": float(item[metric]),
        }
        for item in history
    ]
    latest_actual = float(history[-1][metric])
    final_value = float(forecast_values[-1]["predicted_value"])
    projected_change = 0.0 if latest_actual == 0 else (final_value - latest_actual) / latest_actual
    trend_direction = _describe_trend(projected_change)
    horizon_text = f"{first_year}" if first_year == final_year else f"{first_year}-{final_year}"
    sector_label = sector_summary_label(sector)

    values_text = ", ".join(
        f"{item['year']}: {item['predicted_value']:.2f}" for item in forecast_values
    )
    summary = (
        f"{sector_label} forecast indicates an {trend_direction} trend for {resolved_country} {metric} across "
        f"horizon {horizon_text}. The latest observed {metric} is {latest_actual:.2f}, and the projected values "
        f"are {values_text}. Forecast method: {forecast_result['method']}. {confidence['reason']}"
    )

    insight = build_insight(
        country=resolved_country,
        score=float(projected_change),
        summary=summary,
        confidence=confidence["score"],
    )
    insight["trend"] = trend_direction
    insight["metric"] = metric
    insight["periods"] = periods
    insight["forecast_values"] = forecast_values
    insight["history_values"] = history_values
    insight["latest_actual"] = latest_actual
    insight["sector"] = sector
    insight["method"] = forecast_result["method"]
    insight["forecast_model_scores"] = forecast_result.get("model_scores", {})
    insight["confidence_reason"] = confidence["reason"]
    insight["confidence_components"] = confidence["components"]
    return [insight]



def _resolve_snapshot_year(data_path: str, year: object) -> int:
    """Resolve the requested year or fall back to the latest available year."""
    if year is None:
        return latest_year(data_path)
    return int(year)



def _describe_trend(projected_change: float) -> str:
    """Convert a projected change score into a simple trend label."""
    if projected_change > 0.01:
        return "increasing"
    if projected_change < -0.01:
        return "decreasing"
    return "stable"

