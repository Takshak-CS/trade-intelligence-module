"""Main execution entry point for the Trade Graph Intelligence Agent.

One entry point, ``execute(query)``, routes a dictionary query to the right
analysis and returns the standard agent envelope described in
``core.output_formatter``. Six query types are supported:

    risk        which countries are structurally exposed in the trade network
    shock       what happens downstream when one country's exports are cut
    forecast    where a country's trade is heading
    leverage    who depends on whom, and who could walk away
    blocs       which countries cluster into trading blocs
    fragility   which sectors a country cannot easily substitute out of

``risk``, ``leverage``, ``blocs``, and ``fragility`` all accept an optional
country. With one, they return that country's profile; without one, they return
a ranked view of the whole network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from core import baci_cache
from core.community import detect_trade_blocs, inter_bloc_flows, summarize_blocs
from core.confidence import build_confidence_assessment, combine_forecast_confidence_assessment
from core.data_loader import (
    cache_directory,
    cache_ready,
    can_resolve_countries,
    latest_year,
    load_country_time_series,
    load_trade_data,
    resolve_baci_country,
)
from core.feature_engineering import compute_country_features
from core.forecast import forecast_metric_from_series
from core.fragility import compute_sector_fragility, fragility_profile, rank_sector_fragility
from core.graph_builder import build_trade_graph
from core.leverage import compute_country_leverage, compute_leverage_pairs, leverage_profile
from core.metadata import (
    attach_economic_exposure,
    describe_exposure,
    load_metadata,
    metadata_available,
)
from core.output_formatter import build_insight, build_metadata, format_agent_output
from core.risk_analyzer import analyze_trade_risk
from core.sector_mapper import SUPPORTED_SECTORS, normalize_sector, sector_summary_label
from core.shock_simulator import simulate_trade_shock


DEFAULT_DATA_PATH = "dataset" if Path("dataset").exists() else "trade_data.csv"
FORECAST_METRICS = {"imports", "exports"}
QUERY_TYPES = ("risk", "shock", "forecast", "leverage", "blocs", "fragility")

# Sector graphs other than "all", used by fragility comparisons.
COMPARABLE_SECTORS = tuple(sector for sector in SUPPORTED_SECTORS if sector != "all")

# Trade below this value (BACI thousands of USD) is excluded from network-wide
# leverage and fragility rankings, so micro-territories do not crowd out
# economically significant findings. Country-specific queries ignore it.
RANKING_TRADE_FLOOR = 1_000_000.0



def execute(query: dict) -> dict:
    """Execute a trade intelligence query and return structured insights."""
    if not isinstance(query, dict):
        raise ValueError("Query must be a dictionary.")

    query_type = str(query.get("query_type", "risk")).strip().lower()
    if query_type not in QUERY_TYPES:
        supported = ", ".join(QUERY_TYPES)
        raise ValueError(f"Unsupported query_type '{query_type}'. Supported values: {supported}")

    data_path = str(query.get("data_path", DEFAULT_DATA_PATH))
    limit = max(1, int(query.get("limit", 10)))
    sector = normalize_sector(query.get("sector", "all"))

    if query_type == "forecast":
        return _execute_forecast_query(data_path, query, sector=sector)

    snapshot_year = _resolve_snapshot_year(data_path, query.get("year"))
    country = _resolve_country_name(data_path, query.get("country"))

    if query_type == "fragility":
        return _execute_fragility_query(data_path, snapshot_year, country, limit)

    snapshot_data = load_trade_data(data_path, year=snapshot_year, sector=sector)
    graph = build_trade_graph(snapshot_data, sector=sector)
    data_quality = snapshot_data.attrs.get("data_quality")

    if query_type == "blocs":
        return _execute_blocs_query(graph, snapshot_year, country, limit, sector, data_quality)

    if query_type == "leverage":
        return _execute_leverage_query(graph, snapshot_year, country, limit, sector, data_quality)

    features = compute_country_features(graph)
    risk = analyze_trade_risk(
        graph,
        top_n_partners=max(1, int(query.get("top_n_partners", 3))),
        centrality=_precomputed_centrality(snapshot_year, sector),
    )

    if query_type == "shock":
        if not country:
            raise ValueError("Shock queries must include a 'country'.")
        return _execute_shock_query(
            graph, features, risk, query, country, snapshot_year, limit, sector, data_quality
        )

    return _execute_risk_query(
        graph, features, risk, country, snapshot_year, limit, sector, data_quality
    )



def _execute_risk_query(
    graph: nx.DiGraph,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    country: Optional[str],
    snapshot_year: int,
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> dict:
    """Rank the network by risk, or profile one country's position in it."""
    merged = risk.merge(features, on="country", how="left")
    sector_label = sector_summary_label(sector)

    if country:
        selected = merged[merged["country"] == country]
        if selected.empty:
            raise ValueError(f"Country '{country}' is not present in the {snapshot_year} trade graph.")
        rank = int(merged.index[merged["country"] == country][0]) + 1
        rows = selected
    else:
        rank = None
        rows = merged.head(limit)

    insights: List[dict] = []
    for offset, row in enumerate(rows.itertuples(index=False)):
        confidence = build_confidence_assessment(
            data_quality=data_quality,
            graph=graph,
            country=row.country,
            propagation_steps=1,
        )
        position = rank if rank is not None else offset + 1
        summary = (
            f"{sector_label} in year {snapshot_year}: {row.country} ranks {position} of "
            f"{len(merged)} by risk score {row.risk_score:.3f}. Its top-{len(row.top_partners)} "
            f"partner share is {row.trade_concentration:.2%} across {row.partner_count} partners "
            f"(HHI {row.concentration_index:.3f}), with key partners "
            f"{format_partner_list(row.top_partners)}. {confidence['reason']}"
        )

        insight = build_insight(
            country=row.country,
            score=float(row.risk_score),
            summary=summary,
            confidence=confidence["score"],
        )
        insight.update(
            {
                "rank": position,
                "sector": sector,
                "pagerank_centrality": float(row.pagerank_centrality),
                "betweenness_centrality": float(row.betweenness_centrality),
                "trade_concentration": float(row.trade_concentration),
                "concentration_index": float(row.concentration_index),
                "partner_count": int(row.partner_count),
                "top_partners": list(row.top_partners),
                "total_imports": _safe_float(getattr(row, "total_imports", None)),
                "total_exports": _safe_float(getattr(row, "total_exports", None)),
                "import_dependency_ratio": _safe_float(getattr(row, "import_dependency_ratio", None)),
                "export_dependency_ratio": _safe_float(getattr(row, "export_dependency_ratio", None)),
                "confidence_reason": confidence["reason"],
                "confidence_components": confidence["components"],
            }
        )
        insights.append(insight)

    return format_agent_output(
        insights,
        metadata=build_metadata(
            "risk",
            sector=sector,
            year=snapshot_year,
            country=country,
            method="pagerank + betweenness + partner concentration (HHI)",
            countries_ranked=len(merged),
        ),
    )



def _execute_shock_query(
    graph: nx.DiGraph,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    query: dict,
    country: str,
    snapshot_year: int,
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> dict:
    """Run a shock simulation and convert it into ranked insights."""
    severity = float(query.get("severity", 0.5))
    steps = max(1, int(query.get("steps", 3)))
    propagation_factor = float(query.get("propagation_factor", 0.7))
    sector_label = sector_summary_label(sector)

    impact = simulate_trade_shock(
        graph=graph,
        country=country,
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
    merged = _enrich_with_economics(merged, features, snapshot_year)

    insights: List[dict] = []
    for row in merged.head(limit).itertuples(index=False):
        confidence = build_confidence_assessment(
            data_quality=data_quality,
            graph=graph,
            country=row.country,
            propagation_steps=propagation_steps,
        )
        gdp_exposure = _safe_float(getattr(row, "gdp_exposure", None))
        economic_clause = ""
        if gdp_exposure is not None:
            economic_clause = (
                f" That is roughly {gdp_exposure:.1%} of national output "
                f"({describe_exposure(gdp_exposure)})."
            )

        summary = (
            f"{sector_label} shock from {country} in year {snapshot_year} produces an estimated "
            f"impact score of {row.impact_score:.2%} for {row.country} after {propagation_steps} "
            f"propagation steps.{economic_clause} {confidence['reason']}"
        )

        insight = build_insight(
            country=row.country,
            score=float(row.impact_score),
            summary=summary,
            confidence=confidence["score"],
        )
        insight.update(
            {
                "sector": sector,
                "impact_score": float(row.impact_score),
                "applied_export_reduction": float(row.applied_export_reduction),
                "risk_score": float(row.risk_score),
                "gdp_exposure": gdp_exposure,
                "disrupted_trade_usd": _safe_float(getattr(row, "disrupted_trade_usd", None)),
                "confidence_reason": confidence["reason"],
                "confidence_components": confidence["components"],
            }
        )
        insights.append(insight)

    return format_agent_output(
        insights,
        metadata=build_metadata(
            "shock",
            sector=sector,
            year=snapshot_year,
            country=country,
            method="iterative export-reduction propagation",
            severity=severity,
            propagation_steps=propagation_steps,
            requested_steps=steps,
            propagation_factor=propagation_factor,
            economics_available=_economics_available(),
        ),
    )



def _execute_leverage_query(
    graph: nx.DiGraph,
    snapshot_year: int,
    country: Optional[str],
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> dict:
    """Rank the most lopsided trade relationships, or profile one country."""
    sector_label = sector_summary_label(sector)

    if country:
        profile = leverage_profile(graph, country, limit=limit)
        position = profile["position"]
        confidence = build_confidence_assessment(
            data_quality=data_quality, graph=graph, country=country, propagation_steps=1
        )

        dominated = position["partners_dominated"]
        dependent = position["partners_dependent_on"]
        critical = position["most_critical_partner"]
        summary = (
            f"{sector_label} in year {snapshot_year}: {country} holds leverage over {dominated} "
            f"partners and is the more dependent party with {dependent}. Its most critical single "
            f"relationship is {critical}, carrying "
            f"{position['critical_partner_dependence']:.1%} of its total trade. "
            f"Net leverage position {position['net_leverage']:.2f}. {confidence['reason']}"
        )

        insight = build_insight(
            country=country,
            score=float(position["net_leverage"]),
            summary=summary,
            confidence=confidence["score"],
        )
        insight.update(
            {
                "sector": sector,
                "leverage_held": float(position["leverage_held"]),
                "leverage_exposed": float(position["leverage_exposed"]),
                "net_leverage": float(position["net_leverage"]),
                "partners_dominated": int(dominated),
                "partners_dependent_on": int(dependent),
                "most_critical_partner": critical,
                "critical_partner_dependence": float(position["critical_partner_dependence"]),
                "holds_leverage_over": profile["holds_leverage_over"],
                "vulnerable_to": profile["vulnerable_to"],
                "confidence_reason": confidence["reason"],
                "confidence_components": confidence["components"],
            }
        )
        insights = [insight]
        pair_count = None
    else:
        pairs = compute_leverage_pairs(graph, min_bilateral_trade=RANKING_TRADE_FLOOR)
        pair_count = len(pairs)
        insights = []
        for row in pairs.head(limit).itertuples(index=False):
            confidence = build_confidence_assessment(
                data_quality=data_quality,
                graph=graph,
                country=row.exposed_country,
                propagation_steps=1,
            )
            summary = (
                f"{sector_label} in year {snapshot_year}: {row.exposed_country} routes "
                f"{row.exposed_dependence:.1%} of its trade through {row.leverage_holder}, which "
                f"routes only {row.holder_dependence:.2%} of its own through {row.exposed_country} "
                f"- a {row.exposure_ratio:.0f}x imbalance. {row.leverage_holder} could absorb a "
                f"rupture that {row.exposed_country} could not. {confidence['reason']}"
            )
            insight = build_insight(
                country=row.exposed_country,
                score=float(row.asymmetry),
                summary=summary,
                confidence=confidence["score"],
            )
            insight.update(
                {
                    "sector": sector,
                    "leverage_holder": row.leverage_holder,
                    "exposed_country": row.exposed_country,
                    "bilateral_trade": float(row.bilateral_trade),
                    "holder_dependence": float(row.holder_dependence),
                    "exposed_dependence": float(row.exposed_dependence),
                    "asymmetry": float(row.asymmetry),
                    "exposure_ratio": float(row.exposure_ratio),
                    "confidence_reason": confidence["reason"],
                    "confidence_components": confidence["components"],
                }
            )
            insights.append(insight)

    return format_agent_output(
        insights,
        metadata=build_metadata(
            "leverage",
            sector=sector,
            year=snapshot_year,
            country=country,
            method="bilateral dependence asymmetry",
            pairs_evaluated=pair_count,
            minimum_bilateral_trade=None if country else RANKING_TRADE_FLOOR,
        ),
    )



def _execute_blocs_query(
    graph: nx.DiGraph,
    snapshot_year: int,
    country: Optional[str],
    limit: int,
    sector: str,
    data_quality: dict | None,
) -> dict:
    """Detect trade blocs, or report which bloc one country belongs to."""
    assignment = detect_trade_blocs(graph)
    summary_frame = summarize_blocs(assignment)
    modularity = float(assignment.attrs.get("modularity", 0.0))
    sector_label = sector_summary_label(sector)

    world_trade = float(summary_frame["total_trade"].sum()) or 1.0

    if country:
        selected = assignment[assignment["country"] == country]
        if selected.empty:
            raise ValueError(f"Country '{country}' is not present in the {snapshot_year} trade graph.")
        row = selected.iloc[0]
        bloc = summary_frame[summary_frame["bloc_id"] == row["bloc_id"]].iloc[0]
        bloc_world_share = float(bloc["total_trade"]) / world_trade
        confidence = build_confidence_assessment(
            data_quality=data_quality, graph=graph, country=country, propagation_steps=1
        )

        summary = (
            f"{sector_label} in year {snapshot_year}: {country} belongs to the {row['bloc_name']} "
            f"bloc, one of {int(assignment.attrs.get('bloc_count', 0))} detected, with "
            f"{int(row['bloc_size'])} members and {bloc_world_share:.1%} of world trade activity. "
            f"{row['internal_trade_share']:.1%} of {country}'s own trade stays inside that bloc. "
            f"Largest members: {format_partner_list(bloc['members'])}. {confidence['reason']}"
        )
        insight = build_insight(
            country=country,
            score=float(row["internal_trade_share"]),
            summary=summary,
            confidence=confidence["score"],
        )
        insight.update(
            {
                "sector": sector,
                "bloc_id": int(row["bloc_id"]),
                "bloc_name": str(row["bloc_name"]),
                "bloc_size": int(row["bloc_size"]),
                "internal_trade_share": float(row["internal_trade_share"]),
                "world_trade_share": bloc_world_share,
                "bloc_members": list(bloc["all_members"]),
                "confidence_reason": confidence["reason"],
                "confidence_components": confidence["components"],
            }
        )
        insights = [insight]
    else:
        insights = []
        for row in summary_frame.head(limit).itertuples(index=False):
            confidence = build_confidence_assessment(
                data_quality=data_quality,
                graph=graph,
                country=row.bloc_name,
                propagation_steps=1,
            )
            share = float(row.total_trade) / world_trade
            summary = (
                f"{sector_label} in year {snapshot_year}: the {row.bloc_name} bloc has "
                f"{row.member_count} members and accounts for {share:.1%} of world trade activity. "
                f"On average {row.mean_internal_share:.1%} of a member's trade stays inside the "
                f"bloc. Largest members: {format_partner_list(row.members)}. {confidence['reason']}"
            )
            insight = build_insight(
                country=row.bloc_name,
                score=share,
                summary=summary,
                confidence=confidence["score"],
            )
            insight.update(
                {
                    "sector": sector,
                    "bloc_id": int(row.bloc_id),
                    "bloc_name": str(row.bloc_name),
                    "member_count": int(row.member_count),
                    "members": list(row.members),
                    "bloc_members": list(row.all_members),
                    "world_trade_share": share,
                    "mean_internal_share": float(row.mean_internal_share),
                    "confidence_reason": confidence["reason"],
                    "confidence_components": confidence["components"],
                }
            )
            insights.append(insight)

    flows = inter_bloc_flows(graph, assignment)
    return format_agent_output(
        insights,
        metadata=build_metadata(
            "blocs",
            sector=sector,
            year=snapshot_year,
            country=country,
            method="Louvain community detection on the undirected trade projection",
            bloc_count=int(assignment.attrs.get("bloc_count", 0)),
            modularity=modularity,
            inter_bloc_flows=flows.head(12).to_dict(orient="records"),
        ),
    )



def _execute_fragility_query(
    data_path: str,
    snapshot_year: int,
    country: Optional[str],
    limit: int,
) -> dict:
    """Compare sectors to find where supply is hardest to substitute."""
    graphs: Dict[str, nx.DiGraph] = {}
    for sector_name in COMPARABLE_SECTORS:
        sector_data = load_trade_data(data_path, year=snapshot_year, sector=sector_name)
        graphs[sector_name] = build_trade_graph(sector_data, sector=sector_name)

    sector_ranking = rank_sector_fragility(graphs)

    if country:
        profile = fragility_profile(graphs, country)
        insights = []
        for record in profile["sectors"]:
            summary = (
                f"{country} draws {record['sector']} imports from {record['supplier_count']} "
                f"suppliers with concentration {record['supplier_concentration']:.3f}. Its largest "
                f"supplier is {record['top_supplier']} at {record['top_supplier_share']:.1%}, and "
                f"{record['sector']} accounts for {record['sector_reliance']:.1%} of the imports "
                f"visible to this module. Fragility score {record['fragility_score']:.3f}."
            )
            insight = build_insight(
                country=country,
                score=float(record["fragility_score"]),
                summary=summary,
                confidence=_fragility_confidence(record),
            )
            insight.update({key: value for key, value in record.items() if key != "country"})
            insights.append(insight)
    else:
        fragility = compute_sector_fragility(graphs, min_sector_imports=RANKING_TRADE_FLOOR)
        insights = []
        for record in fragility.head(limit).to_dict(orient="records"):
            summary = (
                f"{record['country']} is most exposed in {record['sector']}: "
                f"{record['supplier_count']} suppliers, concentration "
                f"{record['supplier_concentration']:.3f}, largest supplier "
                f"{record['top_supplier']} at {record['top_supplier_share']:.1%}. That sector is "
                f"{record['sector_reliance']:.1%} of its visible imports. Fragility score "
                f"{record['fragility_score']:.3f}."
            )
            insight = build_insight(
                country=str(record["country"]),
                score=float(record["fragility_score"]),
                summary=summary,
                confidence=_fragility_confidence(record),
            )
            insight.update({key: value for key, value in record.items() if key != "country"})
            insights.append(insight)

    return format_agent_output(
        insights,
        metadata=build_metadata(
            "fragility",
            sector="all",
            year=snapshot_year,
            country=country,
            method="supplier concentration weighted with sector reliance",
            sectors_compared=list(COMPARABLE_SECTORS),
            sector_supply_concentration=sector_ranking.to_dict(orient="records"),
            minimum_sector_imports=None if country else RANKING_TRADE_FLOOR,
        ),
    )



def _execute_forecast_query(data_path: str, query: dict, sector: str) -> dict:
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
    heuristic_confidence = build_confidence_assessment(
        data_quality=time_series_payload.get("data_quality"),
        graph=latest_graph,
        country=resolved_country,
        propagation_steps=1,
    )
    confidence = combine_forecast_confidence_assessment(
        heuristic_assessment=heuristic_confidence,
        model_confidence=float(forecast_result.get("model_confidence", forecast_result.get("confidence", 0.0))),
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
    insight["model_confidence"] = float(
        forecast_result.get("model_confidence", forecast_result.get("confidence", 0.0))
    )
    insight["heuristic_confidence"] = float(heuristic_confidence["score"])
    insight["confidence_reason"] = confidence["reason"]
    insight["confidence_components"] = confidence["components"]

    return format_agent_output(
        [insight],
        metadata=build_metadata(
            "forecast",
            sector=sector,
            country=resolved_country,
            method=str(forecast_result["method"]),
            metric=metric,
            periods=periods,
            horizon=horizon_text,
            history_years=len(history_values),
            model_scores=forecast_result.get("model_scores", {}),
        ),
    )



def _resolve_snapshot_year(data_path: str, year: object) -> int:
    """Resolve the requested year or fall back to the latest available year."""
    if year is None:
        return latest_year(data_path)
    return int(year)



def _resolve_country_name(data_path: str, country: object) -> Optional[str]:
    """Turn a name, ISO code, or numeric code into the canonical country name.

    Returns None when no country was supplied, which the ranked query paths
    treat as "score the whole network".
    """
    if not country:
        return None
    if not can_resolve_countries(data_path):
        return str(country)
    _, resolved = resolve_baci_country(data_path, str(country))
    return resolved



def _precomputed_centrality(year: int, sector: str):
    """Fetch precomputed centrality when the cache has it, else None.

    Returning None is not an error: the risk analyzer falls back to computing
    centrality live, which is what happens on a cache-free deployment.
    """
    if not cache_ready():
        return None
    return baci_cache.load_centrality(cache_directory(), year=year, sector=sector)



def _economics_available() -> bool:
    """Return True when GDP and population enrichment can be applied."""
    return cache_ready() and metadata_available(cache_directory())



def _enrich_with_economics(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Add GDP-relative exposure when country metadata has been fetched."""
    if not _economics_available():
        return frame

    try:
        cache_dir = cache_directory()
        return attach_economic_exposure(
            impact=frame,
            features=features,
            metadata=load_metadata(cache_dir),
            country_codes=baci_cache.load_countries(cache_dir),
            year=year,
        )
    except (ValueError, KeyError):
        # Enrichment is a bonus, never a reason to fail a query.
        return frame



def _fragility_confidence(record: dict) -> float:
    """Confidence in a fragility score, driven by how much evidence supports it.

    A score computed from three supplier relationships is a weaker claim than
    one computed from eighty, so supplier count is the dominant term.
    """
    supplier_count = int(record.get("supplier_count", 0))
    coverage = min(1.0, supplier_count / 20.0)
    global_pool = int(record.get("global_supplier_pool", 0))
    breadth = min(1.0, global_pool / 100.0) if global_pool else 0.5
    return float(max(0.1, min(0.95, 0.35 + (0.45 * coverage) + (0.2 * breadth))))



def _safe_float(value: object) -> Optional[float]:
    """Convert a value to float, returning None for missing or unusable input."""
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(converted) else converted



def format_partner_list(partners: object) -> str:
    """Render a partner list for prose.

    Several BACI country names contain commas ("China, Hong Kong SAR"), so a
    comma-joined list is ambiguous to read. Semicolons cannot occur in the
    names and keep the boundaries unmistakable.
    """
    if partners is None:
        return "no major partners"
    items = [str(partner).strip() for partner in partners if str(partner).strip()]
    if not items:
        return "no major partners"
    return "; ".join(items)



def _describe_trend(projected_change: float) -> str:
    """Convert a projected change score into a simple trend label."""
    if projected_change > 0.01:
        return "increasing"
    if projected_change < -0.01:
        return "decreasing"
    return "stable"
