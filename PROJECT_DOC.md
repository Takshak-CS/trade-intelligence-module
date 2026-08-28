# Project Documentation

Trade Graph Intelligence Agent — module 3 of the Team 128 multi-agent
geopolitical intelligence system.

## 1. System Architecture

### Entry point

`trade_agent.execute(query)` in `agent/trade_agent.py` is the single entry
point. It validates the query, resolves the country and year, loads the right
data slice, runs the requested analysis, and returns a standard envelope. The
FastAPI layer in `api/app.py` is a thin wrapper over it, so the agent is equally
usable as a library import or as an HTTP service.

### Layers

```text
            HTTP                     agent                    core
  frontend ──► api/app.py ──► agent/trade_agent.py ──► core/*.py ──► cache/*.parquet
                                                                          ▲
                                                    scripts/build_cache.py ┘
                                                          (offline, one-time)
```

The important structural decision is that **nothing at request time reads the
raw dataset**. A one-time offline pass converts 8.2 GB of product-level CSVs
into 21 MB of country-level parquet, and every query is a filter over that.

### Modules

| Module | Responsibility |
|--------|----------------|
| `core/baci_cache.py` | Reads the parquet cache; thread-safe bounded in-process memo |
| `core/data_loader.py` | Source resolution and cleaning: cache, then BACI CSVs, then a normalized CSV |
| `core/graph_builder.py` | Trade table to directed weighted NetworkX graph |
| `core/feature_engineering.py` | Imports, exports, dependency ratios |
| `core/risk_analyzer.py` | Centrality, HHI concentration, composite risk |
| `core/shock_simulator.py` | Iterative export-shock propagation |
| `core/leverage.py` | Bilateral dependence asymmetry |
| `core/community.py` | Louvain trade blocs and bloc evolution |
| `core/fragility.py` | Cross-sector supplier concentration |
| `core/forecast.py` | Linear / ARIMA / hybrid with rolling-origin model selection |
| `core/metadata.py` | Optional GDP and population enrichment |
| `core/confidence.py` | Interpretable confidence scoring |
| `core/sector_mapper.py` | HS92 chapter to sector mapping |
| `core/output_formatter.py` | The shared response envelope |

## 2. Data Pipeline

### Preprocessing

BACI ships one CSV per year, roughly 10.6 million product-level rows each with
columns `t, i, j, k, v, q` (year, exporter, importer, HS92 product, value,
quantity). `scripts/build_cache.py` makes one pass per file and:

1. Labels each row's HS92 chapter as agriculture (01–24), energy (27),
   electronics (84, 85, 90), or other.
2. Aggregates to exporter-importer-sector totals, deriving the `all` sector by
   summing across labels rather than re-reading the file.
3. Maps numeric country codes to names, drops self-trade and negative values.
4. Measures field completeness for the confidence model.
5. Precomputes PageRank and betweenness per year and sector.

### Cache layout

```text
cache/
  edges/edges_<year>.parquet   exporter, importer, year, sector, trade_value
  country_year.parquet         per-country yearly totals and ratios
  centrality.parquet           PageRank and betweenness per year and sector
  quality.parquet              completeness per year and sector
  countries.parquet            bundled country code table
  country_metadata.parquet     optional: World Bank GDP and population
  meta.json                    build manifest
```

The cache bundles the country code table so it works without the raw dataset.

### Measured effect

| Query | Raw CSVs | Cache | Speedup |
|-------|---------:|------:|--------:|
| risk | 19.1s | 0.25s | 76x |
| shock | 7.0s | 0.43s | 16x |
| forecast (cold) | 314s | 0.19s | 1650x |

Build time is about 12 minutes once, producing 21 MB.

## 3. Analytical Methods

### Graph construction

Nodes are countries, directed edges are exports weighted by trade value.
Product detail is aggregated away before construction, so analysis is
country-level within a sector.

### Risk

A weighted combination of PageRank, betweenness centrality computed over an
inverted-weight distance graph (so high-value routes are short paths), and
partner concentration.

Concentration uses the **Herfindahl-Hirschman Index** over the full partner
distribution rather than a top-N partner share. The top-N share saturates at
1.0 for any country with N or fewer partners, which made small countries on
sector subgraphs look maximally concentrated regardless of their actual
position. HHI degrades gracefully at any partner count.

### Shock propagation

1. Reduce the chosen country's outgoing edges by the severity.
2. Measure each downstream country's import loss against baseline.
3. Convert that loss into a secondary export reduction, scaled by the
   propagation factor.
4. Repeat until nothing changes or the step budget is spent.

The simulation copies the graph, so repeated runs are independent. It is a
propagation heuristic, not a general-equilibrium model.

### Leverage

For each country pair, dependence is the share of each side's total trade that
the relationship represents. The side with the lower share holds the leverage:
it gives up less if the relationship breaks. Aggregating across a country's
relationships gives its net position.

Example output for 2024: North Korea routes 92.2% of its trade through China,
which routes 0.04% of its own through North Korea — a 2462x imbalance.

### Trade blocs

Louvain community detection over the undirected projection of the trade graph,
where each edge carries the two-way value of the relationship. The seed is
fixed so results are reproducible; blocs are ordered by economic weight and
named after their largest member.

2024 result: three blocs — China (125 members, 47.0% of world trade), Germany
(64, 34.6%), USA (37, 18.4%).

### Sector fragility

A country is fragile in a sector when it leans on imports there **and** those
imports come from few suppliers. The score weights supplier concentration (HHI)
at 0.6 and sector reliance at 0.4. A separate network-level ranking measures how
concentrated global supply is per sector, which indicates how hard substitution
would be.

2024 result: electronics is the most concentrated sector globally (HHI 0.096,
China 25.9% of world supply).

### Forecasting

Three models are backtested with rolling-origin MAPE and the winner is used:
linear trend, ARIMA(1,1,1), and a hybrid of linear trend plus ARIMA-modelled
residuals.

**ARIMA rather than SARIMA is deliberate.** The series is annual, so there is no
within-year seasonality for a seasonal term to capture; adding one would fit
noise. For Chinese exports over 1995–2024 the selector compared hybrid 0.0521,
linear 0.0576, and ARIMA 0.0754, and chose the hybrid.

### Confidence

Every insight carries a confidence score in [0, 1] and the name of the factor
that limited it. Three components:

- **Data completeness** — field availability in the underlying slice. When it
  cannot be assessed the score is neutral (0.5), not perfect, so the module
  never asserts completeness it has not measured.
- **Graph connectivity** — the country's degree relative to the network, and
  overall density.
- **Propagation reliability** — falls as a result depends on more inference
  steps.

Forecasts blend a model-fit confidence with the heuristic score.

## 4. Response Contract

Every query type returns the same envelope so an orchestrator can fuse results
across agents without special-casing:

```json
{
  "agent": "trade",
  "metadata": { "query_type": "...", "sector": "...", "year": 2024, "method": "..." },
  "insights": [
    { "country": "...", "score": 0.0, "summary": "...", "confidence": 0.0,
      "confidence_reason": "...", "confidence_components": {} }
  ]
}
```

`GET /capabilities` publishes the query types, parameters, year range, and
country list, so an orchestrator can discover what to route here rather than
being hard-coded against it.

## 5. Design Decisions

**Why graph modelling.** Trade is relational. A graph captures flow direction,
partner concentration, centrality, and cascading effects in one structure.

**Why parquet instead of PostgreSQL.** The HLD named PostgreSQL. This data is
immutable historical panel data, read constantly and never updated — a columnar
file format is the right fit and a row-store transactional database is not.
Parquet plus an in-process memo delivers sub-second queries with no service to
operate. This is a deliberate deviation, not an unfinished task.

**Why no deep learning.** Every score here can be traced to a specific input and
explained in a sentence. That matters more than marginal accuracy for a system
whose output feeds an explanation-producing fusion agent.

**Why country-optional queries.** An orchestrator asks two shapes of question:
"rank the world" and "tell me about X". Risk, leverage, blocs, and fragility all
answer both from the same endpoint.

## 6. Testing

58 tests in `tests/`. Analytics tests run on small hand-built graphs and need no
data; API tests that need real data skip themselves when the cache is absent.

Coverage includes the failure modes that actually bite graph simulations: input
mutation across runs, repeatability, monotonicity in severity, isolated nodes,
pure importers, degenerate single-partner countries, cache-versus-live agreement
on centrality, bloc determinism, and conservation of trade across bloc flows.

## 7. Limitations

- Graph weights are trade value only. Quantity is tracked for data quality but
  does not enter the graph.
- Shock propagation is a heuristic, not a calibrated economic model.
- Product detail is aggregated before graph construction, so analysis is
  country-level within a sector.
- Sector coverage is three HS92 chapter groupings, not a full taxonomy.
- Forecasts are directional signals over annual aggregates.
- GDP enrichment is optional and depends on World Bank coverage, which lags the
  newest BACI year; values are carried forward up to five years and the staleness
  is reported.

## 8. Possible Extensions

- Product-level subgraphs for chokepoint analysis below the sector level.
- Bloc evolution surfaced in the UI — `core.community.track_bloc_evolution`
  already computes it across years.
- Graph neural network embeddings for anomaly detection.
- Exportable scorecards from the dashboard.
