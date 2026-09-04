"""Tests for the HTTP contract.

These focus on the surface an orchestrator depends on: request validation, the
shape of the response envelope, and the discovery endpoints. Tests that need
real trade data are skipped when the parquet cache has not been built, so the
suite stays runnable on a fresh clone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from core.data_loader import cache_ready, coverage_years

client = TestClient(app)

requires_data = pytest.mark.skipif(
    not coverage_years("dataset"),
    reason="No trade data available. Run scripts/build_cache.py first.",
)


def latest_available_year() -> int:
    return max(coverage_years("dataset"))


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_health_reports_readiness():
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["agent"] == "trade"
    assert payload["status"] in {"ok", "degraded"}
    assert isinstance(payload["ready"], bool)
    assert payload["ready"] == (payload["years_available"] > 0)


def test_capabilities_describes_every_query_type():
    response = client.get("/capabilities")
    assert response.status_code == 200

    payload = response.json()
    assert payload["agent"] == "trade"
    assert set(payload["query_types"]) == {
        "risk",
        "shock",
        "forecast",
        "leverage",
        "blocs",
        "fragility",
    }
    assert "response_envelope" in payload
    assert "all" in payload["parameters"]["sector"]


def test_capabilities_query_types_match_the_agent():
    """Discovery must not drift from what the agent actually accepts."""
    from agent.trade_agent import QUERY_TYPES

    advertised = set(client.get("/capabilities").json()["query_types"])
    assert advertised == set(QUERY_TYPES)


def test_openapi_schema_is_served():
    """The orchestrator needs a machine-readable schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/query" in response.json()["paths"]


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------


def test_unknown_query_type_is_rejected():
    response = client.post("/query", json={"query_type": "nonsense"})
    assert response.status_code == 422


def test_shock_without_country_is_rejected():
    response = client.post("/query", json={"query_type": "shock", "severity": 0.5})
    assert response.status_code == 422


def test_shock_without_severity_is_rejected():
    response = client.post("/query", json={"query_type": "shock", "country": "China"})
    assert response.status_code == 422


def test_severity_outside_range_is_rejected():
    response = client.post(
        "/query", json={"query_type": "shock", "country": "China", "severity": 1.5}
    )
    assert response.status_code == 422


def test_forecast_without_country_is_rejected():
    response = client.post("/query", json={"query_type": "forecast"})
    assert response.status_code == 422


def test_limit_is_bounded():
    response = client.post("/query", json={"query_type": "risk", "limit": 5000})
    assert response.status_code == 422


def test_data_path_is_not_accepted_from_callers():
    """A caller must not be able to point the service at an arbitrary file."""
    response = client.post(
        "/query",
        json={"query_type": "risk", "limit": 1, "data_path": "C:/Windows/win.ini"},
    )
    # Rejected as an unknown field. The 422 echoes the caller's own input back,
    # which is standard validation behaviour and not a disclosure -- what
    # matters is that the path never reaches the loader.
    assert response.status_code == 422
    assert "data_path" in response.text
    assert "extra_forbidden" in response.text


def test_misspelled_parameter_is_rejected_not_ignored():
    """A typo must fail loudly rather than silently using the default.

    With Pydantic's default behaviour, "yr" would be dropped and the query
    answered against the latest year -- a plausible but wrong answer, which is
    the worst possible outcome for an orchestrator wiring up to this API.
    """
    response = client.post("/query", json={"query_type": "risk", "yr": 2020, "limit": 1})
    assert response.status_code == 422
    assert "yr" in response.text


def test_unknown_fields_are_rejected():
    response = client.post(
        "/query",
        json={"query_type": "risk", "limit": 1, "evil": True, "__proto__": {"x": 1}},
    )
    assert response.status_code == 422


def test_valid_request_still_accepted_under_strict_extras():
    """Strictness must not break the fields the dashboard actually sends."""
    response = client.post(
        "/query",
        json={
            "query_type": "shock",
            "country": "China",
            "sector": "all",
            "year": 2020,
            "limit": 3,
            "severity": 0.5,
            "steps": 3,
            "top_n_partners": 3,
            "propagation_factor": 0.7,
        },
    )
    assert response.status_code in {200, 400}, response.text


# --------------------------------------------------------------------------
# Envelope shape
# --------------------------------------------------------------------------


def assert_valid_envelope(payload: dict, expected_query_type: str) -> None:
    """Every query type must satisfy the shared four-agent contract.

    The orchestrator fans out to four agents and fuses what comes back, so the
    envelope is the integration surface. A drift here breaks the fusion layer,
    not just this module.
    """
    assert payload["agent"] == "trade_intelligence"
    assert payload["metadata"]["query_type"] == expected_query_type
    assert "data_quality" in payload["metadata"]
    assert isinstance(payload["insights"], list)
    assert payload["insights"], "expected at least one insight"

    for insight in payload["insights"]:
        # entity_iso3 is the join key; it may be None but the field must exist.
        assert "entity_iso3" in insight
        if insight["entity_iso3"] is not None:
            assert isinstance(insight["entity_iso3"], str) and insight["entity_iso3"]
        assert isinstance(insight["entity_name"], str) and insight["entity_name"]
        assert isinstance(insight["claim"], str) and insight["claim"]
        assert isinstance(insight["score"], (int, float))
        assert 0.0 <= insight["confidence"] <= 1.0
        assert isinstance(insight["reason"], str) and insight["reason"]
        assert isinstance(insight["evidence"], dict)


@requires_data
@pytest.mark.parametrize("query_type", ["risk", "leverage", "blocs", "fragility"])
def test_ranked_queries_return_valid_envelopes(query_type):
    response = client.post(
        "/query",
        json={"query_type": query_type, "year": latest_available_year(), "limit": 3},
    )
    assert response.status_code == 200, response.text
    assert_valid_envelope(response.json(), query_type)


@requires_data
@pytest.mark.parametrize("query_type", ["risk", "leverage", "blocs", "fragility"])
def test_country_scoped_queries_return_that_country(query_type):
    """Supplying a country must switch to a profile, not be silently ignored."""
    response = client.post(
        "/query",
        json={"query_type": query_type, "country": "India", "year": latest_available_year()},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert_valid_envelope(payload, query_type)
    assert payload["metadata"]["country"] == "India"
    assert all(insight["entity_name"] == "India" for insight in payload["insights"])
    assert all(insight["entity_iso3"] == "IND" for insight in payload["insights"])


@requires_data
def test_risk_profile_reports_a_rank():
    response = client.post(
        "/query",
        json={"query_type": "risk", "country": "India", "year": latest_available_year()},
    )
    insight = response.json()["insights"][0]
    assert insight["entity_iso3"] == "IND"
    evidence = insight["evidence"]
    assert evidence["rank"] >= 1
    assert evidence["partner_count"] > 0
    assert 0.0 <= evidence["concentration_index"] <= 1.0


@requires_data
def test_shock_returns_bounded_impact_scores():
    response = client.post(
        "/query",
        json={
            "query_type": "shock",
            "country": "China",
            "severity": 0.5,
            "year": latest_available_year(),
            "limit": 5,
        },
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert_valid_envelope(payload, "shock")
    assert all(0.0 <= insight["score"] <= 1.0 for insight in payload["insights"])
    assert payload["metadata"]["severity"] == 0.5


@requires_data
def test_country_can_be_given_as_iso3():
    """Name, ISO2, ISO3, and numeric code must all resolve to the same country."""
    results = []
    for identifier in ["China", "CHN", "CN", "156"]:
        response = client.post(
            "/query",
            json={"query_type": "risk", "country": identifier, "year": latest_available_year()},
        )
        assert response.status_code == 200, f"{identifier}: {response.text}"
        results.append(response.json()["insights"][0]["entity_iso3"])

    assert set(results) == {"CHN"}


@requires_data
def test_unknown_country_returns_a_client_error():
    response = client.post(
        "/query", json={"query_type": "risk", "country": "Atlantis", "year": latest_available_year()}
    )
    assert response.status_code == 400
    assert "Atlantis" in response.json()["detail"]


@requires_data
def test_graph_snapshot_is_capped():
    response = client.get("/graph", params={"year": latest_available_year(), "max_edges": 50})
    assert response.status_code == 200

    payload = response.json()
    assert len(payload["edges"]) <= 50
    assert payload["nodes"]
    assert payload["year"] == latest_available_year()


@requires_data
@pytest.mark.skipif(not cache_ready(), reason="Forecast needs the full cached history.")
def test_forecast_returns_history_and_projection():
    response = client.post(
        "/query",
        json={"query_type": "forecast", "country": "China", "metric": "exports", "periods": 3},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert_valid_envelope(payload, "forecast")

    evidence = payload["insights"][0]["evidence"]
    assert len(evidence["forecast_values"]) == 3
    assert evidence["history_values"]
    assert evidence["trend"] in {"increasing", "decreasing", "stable"}
    assert all(item["predicted_value"] >= 0 for item in evidence["forecast_values"])


# --------------------------------------------------------------------------
# Shared four-agent contract
# --------------------------------------------------------------------------


def test_capabilities_advertises_the_shared_contract():
    """An orchestrator discovers the envelope here rather than reading source."""
    payload = client.get("/capabilities").json()

    assert payload["join_key"] == "entity_iso3"
    envelope = payload["response_envelope"]
    assert envelope["agent"] == "trade_intelligence"
    for field in ("entity_iso3", "entity_name", "claim", "score", "confidence", "reason", "evidence"):
        assert field in envelope["insights"][0], f"{field} not advertised"


def test_agent_name_matches_the_contract():
    """The orchestrator routes on this name; the internal shorthand is not it."""
    from core.output_formatter import AGENT_NAME

    assert AGENT_NAME == "trade_intelligence"


@requires_data
@pytest.mark.parametrize(
    "query_type, payload",
    [
        ("risk", {"query_type": "risk", "limit": 3}),
        ("shock", {"query_type": "shock", "country": "China", "severity": 0.5, "limit": 3}),
        ("forecast", {"query_type": "forecast", "country": "India", "periods": 2}),
        ("leverage", {"query_type": "leverage", "limit": 3}),
        ("blocs", {"query_type": "blocs", "limit": 3}),
        ("fragility", {"query_type": "fragility", "limit": 3}),
    ],
)
def test_every_query_type_satisfies_the_contract(query_type, payload):
    """All six query types must be consumable by the same fusion code."""
    if query_type not in {"forecast"}:
        payload["year"] = latest_available_year()

    response = client.post("/query", json=payload)
    assert response.status_code == 200, response.text
    assert_valid_envelope(response.json(), query_type)


@requires_data
def test_insights_carry_a_resolvable_join_key():
    """Fusion cannot merge an agent's output without ISO3 on the rows."""
    for payload in (
        {"query_type": "risk", "limit": 5},
        {"query_type": "leverage", "limit": 5},
        {"query_type": "fragility", "limit": 5},
    ):
        payload["year"] = latest_available_year()
        insights = client.post("/query", json=payload).json()["insights"]
        resolved = [i for i in insights if i["entity_iso3"]]
        assert len(resolved) == len(insights), (
            f"{payload['query_type']}: {len(insights) - len(resolved)} insights lack an ISO3 code"
        )


@requires_data
def test_evidence_carries_the_numbers_behind_the_claim():
    """A fused briefing has to cite what a claim rests on, not just assert it."""
    response = client.post(
        "/query",
        json={"query_type": "leverage", "year": latest_available_year(), "limit": 1},
    )
    evidence = response.json()["insights"][0]["evidence"]
    for field in ("leverage_holder", "leverage_holder_iso3", "asymmetry", "exposure_ratio"):
        assert field in evidence, f"{field} missing from evidence"
