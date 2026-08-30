"""FastAPI application exposing the trade intelligence agent.

Three surfaces:

    POST /query         run an analysis and get the standard agent envelope
    GET  /graph         a visualization-friendly network snapshot
    GET  /capabilities  what this agent can answer, and over what data
    GET  /health        liveness and readiness for container orchestration

``/capabilities`` exists for the orchestration layer. An orchestrator deciding
whether to route a question here should not have to read this file to find out
which query types, sectors, years, and countries the agent supports.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent import trade_agent
from core import baci_cache
from core.data_loader import cache_directory, cache_ready, coverage_years, latest_year, load_trade_data
from core.graph_builder import build_graph_payload, build_trade_graph
from core.metadata import metadata_available
from core.sector_mapper import SUPPORTED_SECTORS

QueryType = Literal["risk", "shock", "forecast", "leverage", "blocs", "fragility"]
SectorName = Literal["all", "energy", "agriculture", "electronics"]

API_VERSION = "2.0.0"

# Queries that operate on a single-year network snapshot rather than a history.
SNAPSHOT_QUERIES = {"risk", "shock", "leverage", "blocs", "fragility"}

# The frontend is served separately, so the API needs CORS - but a wildcard lets
# any page the user has open reach a locally running instance. Default to the
# origins the bundled frontend actually uses, and let a deployment override it.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def configured_origins() -> list[str]:
    """Allowed CORS origins, overridable with TRADE_CORS_ORIGINS."""
    configured = os.environ.get("TRADE_CORS_ORIGINS", "").strip()
    if not configured:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


class TradeQuery(BaseModel):
    """Validated request body for the trade intelligence endpoint."""

    # Reject unknown fields rather than dropping them. Pydantic's default is to
    # ignore them, which means a caller that misspells a parameter -- "yr"
    # instead of "year" -- gets a successful response computed against the
    # default instead of an error. Across four agents wiring up to this API,
    # a plausible-but-wrong answer is far more expensive than a 422.
    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    country: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Country name, ISO2, ISO3, or BACI numeric code. "
        "For risk, leverage, blocs, and fragility this switches from a ranked "
        "network view to a single-country profile.",
    )
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    severity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=50)
    sector: SectorName = "all"
    metric: Optional[Literal["imports", "exports"]] = None
    forecast_model: Optional[Literal["auto", "linear", "arima", "hybrid"]] = None
    periods: Optional[int] = Field(default=None, ge=1, le=10)
    steps: Optional[int] = Field(default=3, ge=1, le=10)
    top_n_partners: Optional[int] = Field(default=3, ge=1, le=10)
    propagation_factor: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_by_query_type(self) -> "TradeQuery":
        """Apply query-type-specific defaults and validation."""
        if self.query_type == "shock":
            if not self.country:
                raise ValueError("Shock queries require a country.")
            if self.severity is None:
                raise ValueError("Shock queries require a severity between 0 and 1.")

        if self.query_type == "forecast":
            if not self.country:
                raise ValueError("Forecast queries require a country.")
            if self.metric is None:
                self.metric = "exports"
            if self.forecast_model is None:
                self.forecast_model = "auto"
            if self.periods is None:
                self.periods = 3

        return self

    def to_agent_query(self) -> dict:
        """Convert the request model into the agent query dictionary."""
        payload = self.model_dump(exclude_none=True)

        if self.query_type == "forecast":
            payload.pop("year", None)
            payload.pop("severity", None)
            payload.pop("steps", None)
            payload.pop("top_n_partners", None)
            payload.pop("propagation_factor", None)
            payload["metric"] = self.metric or "exports"
            payload["forecast_model"] = self.forecast_model or "auto"
            payload["periods"] = self.periods or 3
        else:
            payload.pop("metric", None)
            payload.pop("forecast_model", None)
            payload.pop("periods", None)
            if self.query_type != "shock":
                payload.pop("severity", None)

        return payload


app = FastAPI(
    title="Trade Graph Intelligence API",
    description=(
        "Graph-based trade intelligence over the CEPII BACI dataset. "
        "Answers risk, shock propagation, forecasting, dependence leverage, "
        "trade bloc, and sector fragility questions."
    ),
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.post("/query")
def query_trade_agent(request: TradeQuery) -> dict:
    """Run a trade analysis query and return structured insights."""
    try:
        return trade_agent.execute(request.to_agent_query())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.get("/graph")
def get_trade_graph(
    year: Optional[int] = Query(default=None, ge=1900, le=2100),
    sector: SectorName = "all",
    max_edges: int = Query(default=140, ge=20, le=400),
) -> dict:
    """Return a visualization-friendly graph snapshot for a selected year and sector."""
    try:
        source_path = trade_agent.DEFAULT_DATA_PATH
        selected_year = latest_year(source_path) if year is None else int(year)
        trade_data = load_trade_data(source_path, year=selected_year, sector=sector)
        graph = build_trade_graph(trade_data, sector=sector)
        payload = build_graph_payload(graph, max_edges=max_edges)
        payload["year"] = selected_year
        payload["sector"] = sector
        return payload
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.get("/health")
def health() -> dict:
    """Liveness and readiness.

    ``ready`` is False when no data source is reachable, which is the state a
    container orchestrator should treat as "do not route traffic here yet".
    """
    years = coverage_years(trade_agent.DEFAULT_DATA_PATH)
    return {
        "status": "ok" if years else "degraded",
        "ready": bool(years),
        "agent": "trade",
        "version": API_VERSION,
        "cache_enabled": cache_ready(),
        "years_available": len(years),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    """Describe what this agent can answer, for orchestrator discovery."""
    data_path = trade_agent.DEFAULT_DATA_PATH
    years = coverage_years(data_path)
    using_cache = cache_ready()

    countries: list[str] = []
    if using_cache:
        try:
            countries = baci_cache.known_countries(cache_directory())
        except (ValueError, OSError):
            countries = []

    return {
        "agent": "trade",
        "version": API_VERSION,
        "description": (
            "Models international trade as a directed weighted graph and answers "
            "questions about structural risk, disruption propagation, dependence "
            "asymmetry, trading blocs, sector substitutability, and trade trends."
        ),
        "response_envelope": {
            "agent": "string",
            "metadata": "object describing the query and the method that answered it",
            "insights": [
                {
                    "country": "string",
                    "score": "float, sortable, meaning depends on query_type",
                    "summary": "string, one self-contained human-readable finding",
                    "confidence": "float 0-1",
                    "confidence_reason": "string explaining the limiting factor",
                }
            ],
        },
        "query_types": {
            "risk": {
                "description": "Rank countries by structural exposure in the trade network.",
                "country": "optional; supplying it returns that country's profile and rank",
                "score_meaning": "composite risk score, higher is more exposed",
            },
            "shock": {
                "description": "Propagate an export disruption and rank affected countries.",
                "country": "required; the country whose exports are disrupted",
                "requires": ["severity"],
                "score_meaning": "share of imports lost, 0-1",
            },
            "forecast": {
                "description": "Project a country's imports or exports forward.",
                "country": "required",
                "score_meaning": "projected fractional change over the horizon",
            },
            "leverage": {
                "description": "Find asymmetric dependence: who needs whom more.",
                "country": "optional; supplying it returns that country's leverage position",
                "score_meaning": "dependence asymmetry, higher means more lopsided",
            },
            "blocs": {
                "description": "Detect trading blocs via Louvain community detection.",
                "country": "optional; supplying it returns that country's bloc membership",
                "score_meaning": "share of world trade for a bloc, or internal trade share for a country",
            },
            "fragility": {
                "description": "Compare sectors to find where supply is hardest to substitute.",
                "country": "optional; supplying it returns that country's sector profile",
                "score_meaning": "fragility score, higher is more brittle",
            },
        },
        "parameters": {
            "sector": list(SUPPORTED_SECTORS),
            "limit": {"min": 1, "max": 50, "default": 5},
            "severity": {"min": 0.0, "max": 1.0},
            "steps": {"min": 1, "max": 10, "default": 3},
            "periods": {"min": 1, "max": 10, "default": 3},
            "forecast_model": ["auto", "linear", "arima", "hybrid"],
            "metric": ["imports", "exports"],
        },
        "coverage": {
            "source": "CEPII BACI HS92",
            "years": years,
            "first_year": min(years) if years else None,
            "last_year": max(years) if years else None,
            "country_count": len(countries),
            "countries": countries,
        },
        "country_identifiers": ["name", "iso2", "iso3", "baci_numeric"],
        "enrichment": {
            "economic_metadata": using_cache and metadata_available(cache_directory()),
            "precomputed_centrality": using_cache
            and baci_cache.centrality_path(cache_directory()).exists(),
        },
    }
