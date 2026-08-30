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
    """Every query type must return the same outer shape."""
    assert payload["agent"] == "trade"
    assert payload["metadata"]["query_type"] == expected_query_type
    assert isinstance(payload["insights"], list)
    assert payload["insights"], "expected at least one insight"

    for insight in payload["insights"]:
        assert isinstance(insight["country"], str) and insight["country"]
        assert isinstance(insight["score"], (int, float))
        assert isinstance(insight["summary"], str) and insight["summary"]
        assert 0.0 <= insight["confidence"] <= 1.0


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
    assert all(insight["country"] == "India" for insight in payload["insights"])


@requires_data
def test_risk_profile_reports_a_rank():
    response = client.post(
        "/query",
        json={"query_type": "risk", "country": "India", "year": latest_available_year()},
    )
    insight = response.json()["insights"][0]
    assert insight["rank"] >= 1
    assert insight["partner_count"] > 0
    assert 0.0 <= insight["concentration_index"] <= 1.0


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
        results.append(response.json()["insights"][0]["country"])

    assert len(set(results)) == 1


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

    insight = payload["insights"][0]
    assert len(insight["forecast_values"]) == 3
    assert insight["history_values"]
    assert insight["trend"] in {"increasing", "decreasing", "stable"}
    assert all(item["predicted_value"] >= 0 for item in insight["forecast_values"])
