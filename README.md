# Trade Graph Intelligence Web Application

## Project Overview

This project models international trade as a directed weighted graph and exposes the analysis through a FastAPI backend and a lightweight HTML and JavaScript frontend. It supports BACI yearly trade files directly from the `dataset/` directory and returns structured trade intelligence outputs for risk analysis, shock simulation, and forecasting.

## Features

- Risk Analysis
  - Builds a country-level trade graph for a selected year.
  - Computes dependency features, PageRank, betweenness, and partner concentration.
  - Returns the top risky countries ranked by a composite score.
- Shock Simulation
  - Applies an export shock to a selected country.
  - Propagates downstream import losses for a few steps.
  - Returns the top affected countries.
- Forecasting
  - Builds a historical country time series from the available BACI files.
  - Uses an interpretable linear trend model.
  - Returns a trend-oriented summary and confidence score.

## Setup Instructions

### 1. Install dependencies

```powershell
pip install pandas numpy networkx statsmodels fastapi uvicorn
```

## Dataset

This project uses the [CEPII BACI International Trade Database](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37).

Download the HS92 dataset and place the yearly CSV files under `dataset/`
in the following structure:

```text
dataset/
|-- BACI_HS92_Y1995_V202601.csv
|-- BACI_HS92_Y1996_V202601.csv
|-- ...
|-- BACI_HS92_Y2024_V202601.csv
|-- country_codes_V202601.csv
`-- product_codes_HS92_V202601.csv
```

The application reads the `dataset/` directory directly. You do not need to merge all years into one CSV.

### 3. Run the backend

From the project root:

```powershell
python -m uvicorn api.app:app --reload
```

The API will start at `http://127.0.0.1:8000`.

### 4. Open the frontend

Serve the frontend directory with a simple static server:

```powershell
python -m http.server 8080 --directory frontend
```

Then open:

```text
http://127.0.0.1:8080
```

You can also open `frontend/index.html` directly, but serving it over HTTP is more reliable.

## Example Queries

### Risk

```json
{
  "query_type": "risk",
  "year": 2020,
  "limit": 5
}
```

### Shock

```json
{
  "query_type": "shock",
  "country": "China",
  "year": 2020,
  "severity": 0.5,
  "limit": 5
}
```

### Forecast

```json
{
  "query_type": "forecast",\n  "country": "CHN",\n  "metric": "exports",\n  "periods": 3
}
```

### cURL example

```powershell
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query_type\":\"risk\",\"year\":2020,\"limit\":5}"
```

## Folder Structure

```text
trade_intelligence_module/
  api/
    app.py
  agent/
    trade_agent.py
  core/
    data_loader.py
    graph_builder.py
    feature_engineering.py
    risk_analyzer.py
    shock_simulator.py
    forecast.py
    output_formatter.py
  dataset/
    ... BACI files ...
  frontend/
    index.html
  PROJECT_DOC.md
  README.md
```

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Pandas
- NumPy
- NetworkX
- Statsmodels
- HTML
- JavaScript
- BACI trade dataset from CEPII


