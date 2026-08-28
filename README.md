# Trade Graph Intelligence

Models international trade as a directed weighted graph and answers structural
questions about it: who is exposed, what happens when a supplier stops, who
depends on whom, which countries cluster together, and where supply cannot be
substituted.

Built on the [CEPII BACI](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)
HS92 dataset — 226 countries, 1995 to 2024, product-level bilateral flows
aggregated to country pairs.

This is the trade intelligence module of a four-agent geopolitical analysis
system (Team 128 capstone). It runs standalone and is designed to be called by
an orchestration agent alongside the soft power, policy stance, and video
intelligence modules.

## What it answers

| Query type  | Question | Score means |
|-------------|----------|-------------|
| `risk`      | Which countries are structurally exposed in the trade network? | Composite risk, higher is more exposed |
| `shock`     | If this country's exports stop, who feels it? | Share of imports lost, 0–1 |
| `forecast`  | Where is this country's trade heading? | Projected fractional change |
| `leverage`  | Who needs whom more? | Dependence asymmetry, higher is more lopsided |
| `blocs`     | Which countries cluster into trading blocs? | Share of world trade, or internal trade share |
| `fragility` | Which sectors can a country not substitute out of? | Fragility score, higher is more brittle |

`risk`, `leverage`, `blocs`, and `fragility` accept an optional `country`. With
one, you get that country's profile. Without one, you get a ranked view of the
whole network.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Download the BACI HS92 release and unpack the yearly CSVs into `dataset/`:

```text
dataset/
  BACI_HS92_Y1995_V202601.csv
  ...
  BACI_HS92_Y2024_V202601.csv
  country_codes_V202601.csv
  product_codes_HS92_V202601.csv
```

The dataset is about 8 GB and is gitignored.

### 3. Build the cache

```bash
python scripts/build_cache.py
```

This makes one pass over each yearly file and writes about 25 MB of parquet into
`cache/`. It takes roughly 15 minutes and only has to happen once.

**This step is not optional in practice.** Without it every query re-parses the
raw CSVs, and a forecast has to touch all thirty files:

| Query | Without cache | With cache |
|-------|--------------:|-----------:|
| risk | 19.1s | 0.25s |
| shock | 7.0s | 0.43s |
| forecast (cold) | 314s | 0.19s |

The cache is self-contained — it bundles the country code table — so a teammate
who has `cache/` but not the 8 GB `dataset/` can run the whole module.

### 4. Optional: fetch economic metadata

```bash
python scripts/fetch_metadata.py
```

Pulls GDP and population from the World Bank so shock results can be expressed
as a share of national output rather than only as a trade share. Everything
works without it; the enrichment is additive.

### 5. Run

```bash
python -m uvicorn api.app:app --reload
```

API at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`.

Then serve the dashboard:

```bash
python -m http.server 8080 --directory frontend
```

Open `http://127.0.0.1:8080`.

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /query` | Run an analysis, get the standard agent envelope |
| `GET /graph` | Visualization-friendly network snapshot |
| `GET /capabilities` | What this agent can answer, over what years and countries |
| `GET /health` | Liveness and readiness |
| `GET /docs` | Interactive OpenAPI documentation |

### Response envelope

Every query type returns the same shape, so an orchestrator can fuse results
without knowing which analysis produced them:

```json
{
  "agent": "trade",
  "metadata": {
    "query_type": "leverage",
    "sector": "all",
    "year": 2024,
    "method": "bilateral dependence asymmetry"
  },
  "insights": [
    {
      "country": "Bhutan",
      "score": 0.7358,
      "summary": "Bhutan routes 73.7% of its trade through India, which routes only 0.13% of its own through Bhutan - a 551x imbalance...",
      "confidence": 0.72,
      "confidence_reason": "Confidence is most sensitive to sparse trade connections for this country."
    }
  ]
}
```

`confidence` is not decoration. Every insight carries a score in [0, 1] plus the
name of the factor that limited it, so a consumer can tell a well-evidenced
finding from a thin one.

### Examples

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_type":"risk","year":2024,"limit":5}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_type":"leverage","year":2024,"limit":10}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_type":"fragility","country":"India","year":2024}'
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_type":"shock","country":"China","severity":0.5,"year":2024,"limit":5}'
```

Country can be a name, ISO2, ISO3, or BACI numeric code — `China`, `CN`, `CHN`,
and `156` all resolve to the same country.

## Docker

```bash
docker build -t trade-intelligence .
docker run -p 8000:8000 -v "$(pwd)/cache:/app/cache:ro" trade-intelligence
```

The 8 GB dataset is deliberately not baked into the image; the cache is mounted
instead.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Analytics tests run on small hand-built graphs and need no data. API tests that
require real trade data skip themselves automatically when the cache is absent.

## Layout

```text
trade_intelligence_module/
  api/app.py               FastAPI surface
  agent/trade_agent.py     query routing and insight construction
  core/
    baci_cache.py          parquet cache reads
    data_loader.py         cache / BACI / CSV loading and cleaning
    graph_builder.py       trade table to NetworkX graph
    feature_engineering.py imports, exports, dependency ratios
    risk_analyzer.py       centrality, HHI concentration, composite risk
    shock_simulator.py     export shock propagation
    leverage.py            asymmetric dependence
    community.py           Louvain trade blocs
    fragility.py           sector substitutability
    forecast.py            linear / ARIMA / hybrid with model selection
    metadata.py            GDP and population enrichment
    confidence.py          interpretable confidence scoring
    sector_mapper.py       HS92 chapter to sector mapping
    output_formatter.py    the shared response envelope
  scripts/
    build_cache.py         one-time parquet precompute
    fetch_metadata.py      World Bank GDP and population
  frontend/index.html      dashboard
  tests/                   pytest suite
```

## Method notes

**Risk** combines PageRank, betweenness centrality computed over an
inverted-weight distance graph, and partner concentration measured as a
Herfindahl-Hirschman Index. HHI is used rather than a top-N partner share
because the share saturates at 1.0 for any country with N or fewer partners,
which misreads small countries on sector subgraphs as maximally concentrated.

**Shock** reduces a country's outgoing edges by the chosen severity, measures the
import loss downstream, converts that into secondary export reductions, and
repeats until nothing changes or the step budget runs out. It is a propagation
heuristic, not a calibrated general-equilibrium model.

**Leverage** measures what share of each side's total trade a bilateral
relationship represents. The country with the lower share holds the leverage: it
gives up less if the relationship breaks.

**Blocs** run Louvain community detection over the undirected projection of the
trade graph, with a fixed seed so results are reproducible.

**Forecast** backtests linear, ARIMA(1,1,1), and a hybrid linear-plus-ARIMA-
residual model using rolling-origin MAPE, then uses the winner. ARIMA rather
than SARIMA is deliberate: the series is annual, so there is no within-year
seasonality for a seasonal term to capture.

**Confidence** combines data completeness, graph connectivity, and propagation
depth, and reports whichever component limited the result. When completeness
cannot be assessed it scores neutral, not perfect.

## Limitations

- Graph weights are trade value only; quantity is tracked for data quality but
  not used in the graph.
- Shock propagation is a heuristic, not an economic equilibrium model.
- Product detail is aggregated to country pairs before graph construction, so
  analysis is country-level within a sector, not product-level.
- Sector coverage is three HS92 chapter groupings (energy, agriculture,
  electronics), not a full sector taxonomy.
- Forecasts are directional signals over annual aggregates, not precise
  macroeconomic predictions.
