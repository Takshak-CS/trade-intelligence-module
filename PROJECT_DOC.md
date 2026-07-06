# Project Documentation

## 1. System Architecture

### Orchestrator

The top-level orchestrator is `trade_agent.execute(query)` in `agent/trade_agent.py`. It validates the query shape, routes the request by query type, loads the appropriate data slice, runs the relevant analysis module, and formats a standardized response.

### Agents

This system currently exposes one domain agent:

- Trade Agent
  - Handles graph construction, risk analysis, shock simulation, and forecast generation.
  - Accepts a dictionary input and returns a structured output payload.

### Trade Pipeline

The pipeline is modular and split into focused components:

- `core/data_loader.py`
  - Detects whether the source is a normalized CSV or a BACI directory.
  - Loads single-year country-to-country trade snapshots.
  - Builds historical country time series for forecasting.
- `core/graph_builder.py`
  - Converts the trade table into a directed NetworkX graph.
- `core/feature_engineering.py`
  - Computes imports, exports, and dependency ratios.
- `core/risk_analyzer.py`
  - Computes centrality, partner concentration, and a composite risk score.
- `core/shock_simulator.py`
  - Applies export shocks and propagates effects iteratively.
- `core/forecast.py`
  - Builds a simple interpretable forecast from historical data.
- `core/output_formatter.py`
  - Wraps the result in the common agent response format.
- `api/app.py`
  - Exposes the pipeline through FastAPI.
- `frontend/index.html`
  - Provides a lightweight browser-based interface.

## 2. Trade Agent Design

### Graph Construction

The system models trade as a directed graph:

- Nodes represent countries.
- Directed edges represent exports from exporter to importer.
- Edge weights currently represent aggregated trade value.

For BACI input, product-level rows are aggregated to exporter-importer-year totals before graph construction.

### Feature Engineering

For each country, the system computes:

- Total imports
- Total exports
- Total trade activity
- Import dependency ratio
- Export dependency ratio

These features provide a basic view of how much a country depends on inbound versus outbound trade activity.

### Risk Model

Risk is modeled as a weighted combination of:

- PageRank centrality
- Betweenness centrality
- Trade concentration among top partners

This design is intentionally interpretable. It surfaces countries that are central in the network, act as trade bridges, or are highly concentrated in a small set of partners.

### Shock Propagation Logic

Shock simulation follows a simple multi-step process:

1. Select a country and severity.
2. Reduce its outgoing edges by the specified severity.
3. Measure the import loss experienced by downstream countries.
4. Convert that loss into secondary export reductions.
5. Propagate for a small number of steps.

The output is an impact score per country, representing how strongly the simulated disruption affects that country.

### Forecasting Logic

Forecasting uses a simple linear trend over historical yearly aggregates for a selected country. The goal is not exact macroeconomic prediction. The goal is a transparent directional signal:

- increasing
- decreasing
- stable

The forecast result is summarized as a projected change over the forecast horizon with an accompanying confidence score.

## 3. Data Flow

The application follows a simple request path:

User -> Frontend -> FastAPI `/query` -> `trade_agent.execute(query)` -> Core pipeline -> Structured JSON response -> Frontend rendering

Detailed sequence:

1. The user selects an analysis type and inputs parameters in the browser.
2. The frontend sends a POST request to `/query`.
3. FastAPI validates the request body.
4. The backend calls the trade agent.
5. The agent loads data, runs the requested analysis, and formats the output.
6. The API returns JSON.
7. The frontend renders both a results table and the raw JSON.

## 4. Design Decisions

### Why graph-based modeling

Trade relationships are inherently relational. A graph model captures:

- direction of trade flow
- partner concentration
- network centrality
- cascading effects from disruptions

This makes graph analysis a natural fit for trade dependency and shock propagation.

### Why no deep learning

The current scope emphasizes interpretability, fast iteration, and lower operational complexity. Classical graph metrics and simple forecasting are easier to inspect, explain, and debug than deep learning models.

### Why modular design

The codebase is organized around small single-purpose modules. This improves:

- readability
- testability
- ability to swap out methods later
- reuse of the trade pipeline in both scripts and the web API

## 5. Limitations

- The current graph is value-based. Quantity is not yet included in graph weights or dependency calculations.
- Shock propagation is a simplified heuristic, not a calibrated economic equilibrium model.
- Forecasting is a basic trend model and should be interpreted as directional guidance, not precise prediction.
- BACI product detail is aggregated before graph construction, so the web app currently operates at the country-country level.

## 6. Future Improvements

- Add quantity-aware graph weights and dependency metrics.
- Add preprocessing and caching for faster repeated queries.
- Add multi-metric risk scoring with configurable weights.
- Add stronger forecasting methods such as ARIMA for selected signals.
- Add sector- or product-specific subgraphs.
- Add graph neural networks for representation learning and anomaly detection.
- Add richer UI controls, saved runs, and downloadable reports.
