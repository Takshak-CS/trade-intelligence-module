"""FastAPI application exposing the trade intelligence agent."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from agent import trade_agent
from core.data_loader import latest_year, load_trade_data
from core.graph_builder import build_graph_payload, build_trade_graph


class TradeQuery(BaseModel):
    """Validated request body for the trade intelligence endpoint."""

    query_type: Literal["risk", "shock", "forecast"]
    country: Optional[str] = None
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    severity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=50)
    data_path: Optional[str] = None
    sector: Literal["all", "energy", "agriculture", "electronics"] = "all"
    metric: Optional[Literal["imports", "exports"]] = None
    forecast_model: Optional[Literal["auto", "linear", "arima", "hybrid"]] = None
    periods: Optional[int] = Field(default=None, ge=1, le=10)
    top_n_partners: Optional[int] = Field(default=None, ge=1, le=10)
    propagation_factor: Optional[float] = Field(default=None, ge=0.0, le=1.0)

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
            payload["metric"] = self.metric or "exports"
            payload["forecast_model"] = self.forecast_model or "auto"
            payload["periods"] = self.periods or 3
        elif self.query_type == "risk":
            payload.pop("severity", None)
            payload.pop("metric", None)
            payload.pop("forecast_model", None)
            payload.pop("periods", None)
        elif self.query_type == "shock":
            payload.pop("metric", None)
            payload.pop("forecast_model", None)
            payload.pop("periods", None)

        return payload


app = FastAPI(
    title="Trade Graph Intelligence API",
    description="FastAPI wrapper for risk, shock, forecast, and trade graph visualization.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
    sector: Literal["all", "energy", "agriculture", "electronics"] = "all",
    max_edges: int = Query(default=140, ge=20, le=400),
    data_path: Optional[str] = None,
) -> dict:
    """Return a visualization-friendly graph snapshot for a selected year and sector."""
    try:
        source_path = data_path or trade_agent.DEFAULT_DATA_PATH
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
