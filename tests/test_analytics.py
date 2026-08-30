"""Tests for the analytical core, run on small hand-built graphs.

Every graph here is small enough to reason about by hand, so a failure points at
a specific broken behaviour rather than "the numbers moved". These cover the
failure modes that actually bite graph simulations: mutating the input, blowing
up on isolated or degenerate nodes, and silently disagreeing with themselves.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from core.community import detect_trade_blocs, inter_bloc_flows, summarize_blocs
from core.confidence import build_confidence_assessment, data_completeness_score
from core.feature_engineering import compute_country_features
from core.fragility import compute_sector_fragility, rank_sector_fragility
from core.leverage import compute_country_leverage, compute_leverage_pairs
from core.risk_analyzer import analyze_trade_risk, compute_trade_concentration
from core.shock_simulator import simulate_trade_shock


@pytest.fixture
def simple_graph() -> nx.DiGraph:
    """A small directed trade graph with a clear hub and a peripheral node."""
    graph = nx.DiGraph()
    for exporter, importer, weight in [
        ("A", "B", 100.0),
        ("B", "C", 50.0),
        ("A", "C", 10.0),
        ("D", "C", 5.0),
        ("C", "A", 20.0),
    ]:
        graph.add_edge(exporter, importer, weight=weight)
    return graph


@pytest.fixture
def two_bloc_graph() -> nx.DiGraph:
    """Two tightly connected clusters joined by a single thin link."""
    graph = nx.DiGraph()
    for exporter, importer, weight in [
        ("A1", "A2", 100.0), ("A2", "A1", 90.0),
        ("A1", "A3", 80.0), ("A3", "A1", 85.0),
        ("A2", "A3", 95.0), ("A3", "A2", 70.0),
        ("B1", "B2", 100.0), ("B2", "B1", 90.0),
        ("B1", "B3", 80.0), ("B3", "B1", 85.0),
        ("B2", "B3", 95.0), ("B3", "B2", 70.0),
        ("A1", "B1", 1.0),
    ]:
        graph.add_edge(exporter, importer, weight=weight)
    return graph


# --------------------------------------------------------------------------
# Shock simulation
# --------------------------------------------------------------------------


def test_shock_does_not_mutate_the_input_graph(simple_graph):
    """Running a simulation must leave the caller's graph untouched."""
    before = {(u, v): d["weight"] for u, v, d in simple_graph.edges(data=True)}
    simulate_trade_shock(simple_graph, "A", severity=1.0, steps=3)
    simulate_trade_shock(simple_graph, "A", severity=0.5, steps=3)
    after = {(u, v): d["weight"] for u, v, d in simple_graph.edges(data=True)}
    assert before == after


def test_shock_is_repeatable(simple_graph):
    """The same query twice must give the same answer."""
    first = simulate_trade_shock(simple_graph, "A", severity=0.6, steps=3)
    second = simulate_trade_shock(simple_graph, "A", severity=0.6, steps=3)
    pd.testing.assert_frame_equal(first, second)


def test_zero_severity_is_a_no_op(simple_graph):
    """A shock of zero must leave every country at zero impact."""
    result = simulate_trade_shock(simple_graph, "A", severity=0.0, steps=3)
    assert result["impact_score"].abs().max() == 0.0


def test_shock_scores_stay_in_range(simple_graph):
    """Impact scores are shares and must never leave [0, 1]."""
    result = simulate_trade_shock(simple_graph, "A", severity=1.0, steps=5)
    assert result["impact_score"].between(0.0, 1.0).all()
    assert result["applied_export_reduction"].between(0.0, 1.0).all()


def test_disconnected_country_is_unaffected(simple_graph):
    """A country with no path from the shocked node takes no impact."""
    simple_graph.add_edge("X", "Y", weight=7.0)
    result = simulate_trade_shock(simple_graph, "A", severity=1.0, steps=3)
    isolated = result[result["country"].isin(["X", "Y"])]
    assert isolated["impact_score"].max() == 0.0


def test_shock_on_pure_importer_does_not_divide_by_zero():
    """A country that never exports has no outgoing edges to reduce."""
    graph = nx.DiGraph()
    graph.add_edge("X", "Y", weight=10.0)
    graph.add_edge("Z", "Y", weight=5.0)
    result = simulate_trade_shock(graph, "Y", severity=1.0, steps=3)
    assert result["impact_score"].notna().all()


def test_greater_severity_never_reduces_impact(simple_graph):
    """Impact must be monotonic in severity."""
    mild = simulate_trade_shock(simple_graph, "A", severity=0.3, steps=3).set_index("country")
    harsh = simulate_trade_shock(simple_graph, "A", severity=0.9, steps=3).set_index("country")
    for country in mild.index:
        assert harsh.loc[country, "impact_score"] >= mild.loc[country, "impact_score"] - 1e-12


def test_unknown_country_is_rejected(simple_graph):
    with pytest.raises(ValueError, match="not present"):
        simulate_trade_shock(simple_graph, "Atlantis", severity=0.5)


# --------------------------------------------------------------------------
# Features and risk
# --------------------------------------------------------------------------


def test_features_match_edge_weights(simple_graph):
    """Imports and exports must sum exactly to the incident edge weights."""
    features = compute_country_features(simple_graph).set_index("country")
    assert features.loc["A", "total_exports"] == pytest.approx(110.0)
    assert features.loc["A", "total_imports"] == pytest.approx(20.0)
    assert features.loc["D", "total_imports"] == pytest.approx(0.0)


def test_dependency_ratios_sum_to_one(simple_graph):
    features = compute_country_features(simple_graph)
    active = features[features["trade_total"] > 0]
    totals = active["import_dependency_ratio"] + active["export_dependency_ratio"]
    assert totals.apply(lambda value: value == pytest.approx(1.0)).all()


def test_concentration_saturates_but_hhi_does_not(simple_graph):
    """The regression this replaced: top-N share maxes out on sparse nodes.

    D has a single partner, so its top-2 share is necessarily 1.0 and tells you
    nothing. The HHI is also 1.0 here, but unlike the top-N share it keeps
    discriminating as partner counts grow, which is why the risk score uses it.
    """
    concentration = compute_trade_concentration(simple_graph, top_n=2).set_index("country")
    assert concentration.loc["D", "trade_concentration"] == pytest.approx(1.0)
    assert concentration.loc["D", "partner_count"] == 1
    assert concentration.loc["C", "trade_concentration"] > concentration.loc["C", "concentration_index"]


def test_hhi_falls_as_partners_multiply():
    """Spreading the same trade over more partners must lower the index."""
    concentrated = nx.DiGraph()
    concentrated.add_edge("hub", "P1", weight=100.0)

    spread = nx.DiGraph()
    for index in range(10):
        spread.add_edge("hub", f"P{index}", weight=10.0)

    tight = compute_trade_concentration(concentrated).set_index("country")
    loose = compute_trade_concentration(spread).set_index("country")
    assert tight.loc["hub", "concentration_index"] > loose.loc["hub", "concentration_index"]
    assert loose.loc["hub", "concentration_index"] == pytest.approx(0.1)


def test_risk_scores_are_ordered(simple_graph):
    risk = analyze_trade_risk(simple_graph, top_n_partners=2)
    assert risk["risk_score"].is_monotonic_decreasing


def test_precomputed_centrality_matches_live_computation(simple_graph):
    """The cache path and the live path must agree exactly."""
    import networkx as network

    from core.risk_analyzer import _build_distance_graph

    live = analyze_trade_risk(simple_graph, top_n_partners=2)
    centrality = pd.DataFrame(
        {
            "country": list(simple_graph.nodes),
            "pagerank_centrality": [
                network.pagerank(simple_graph, weight="weight")[node] for node in simple_graph.nodes
            ],
            "betweenness_centrality": [
                network.betweenness_centrality(
                    _build_distance_graph(simple_graph), weight="weight", normalized=True
                )[node]
                for node in simple_graph.nodes
            ],
        }
    )
    cached = analyze_trade_risk(simple_graph, top_n_partners=2, centrality=centrality)
    pd.testing.assert_frame_equal(live, cached)


def test_empty_graph_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        analyze_trade_risk(nx.DiGraph())


# --------------------------------------------------------------------------
# Leverage
# --------------------------------------------------------------------------


def test_leverage_identifies_the_less_dependent_party():
    """The giant holds leverage over the minnow, not the other way round."""
    graph = nx.DiGraph()
    graph.add_edge("Giant", "Minnow", weight=50.0)
    graph.add_edge("Minnow", "Giant", weight=50.0)
    for index in range(20):
        graph.add_edge("Giant", f"Other{index}", weight=500.0)

    pairs = compute_leverage_pairs(graph)
    relationship = pairs[pairs["exposed_country"] == "Minnow"].iloc[0]
    assert relationship["leverage_holder"] == "Giant"
    assert relationship["exposed_dependence"] > relationship["holder_dependence"]
    assert relationship["asymmetry"] > 0


def test_symmetric_relationship_has_no_leverage():
    """Two identical partners trading only with each other are balanced."""
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=100.0)
    graph.add_edge("B", "A", weight=100.0)

    pairs = compute_leverage_pairs(graph)
    assert pairs["asymmetry"].abs().max() == pytest.approx(0.0)


def test_country_leverage_nets_out(simple_graph):
    summary = compute_country_leverage(simple_graph).set_index("country")
    for country in summary.index:
        expected = summary.loc[country, "leverage_held"] - summary.loc[country, "leverage_exposed"]
        assert summary.loc[country, "net_leverage"] == pytest.approx(expected)


def test_leverage_trade_floor_filters_small_relationships():
    """The floor drops relationships that are lopsided but economically trivial."""
    graph = nx.DiGraph()
    graph.add_edge("Big", "Small", weight=10.0)
    graph.add_edge("Small", "Big", weight=10.0)
    for index in range(10):
        graph.add_edge("Big", f"Other{index}", weight=1000.0)

    unfiltered = compute_leverage_pairs(graph, min_bilateral_trade=0.0)
    assert "Small" in set(unfiltered["exposed_country"])

    # Big<->Small is worth 20 in total; the Other relationships are worth 1000.
    filtered = compute_leverage_pairs(graph, min_bilateral_trade=100.0)
    assert "Small" not in set(filtered["exposed_country"])
    assert len(filtered) == 10

    # A floor above every relationship leaves nothing.
    assert len(compute_leverage_pairs(graph, min_bilateral_trade=10_000.0)) == 0


# --------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------


def test_blocs_separate_two_obvious_clusters(two_bloc_graph):
    """Two dense clusters joined by one thin edge must not merge."""
    assignment = detect_trade_blocs(two_bloc_graph).set_index("country")
    a_blocs = {assignment.loc[node, "bloc_id"] for node in ["A1", "A2", "A3"]}
    b_blocs = {assignment.loc[node, "bloc_id"] for node in ["B1", "B2", "B3"]}
    assert len(a_blocs) == 1
    assert len(b_blocs) == 1
    assert a_blocs != b_blocs


def test_bloc_detection_is_deterministic(two_bloc_graph):
    """A fixed seed means two identical queries give identical blocs."""
    first = detect_trade_blocs(two_bloc_graph)
    second = detect_trade_blocs(two_bloc_graph)
    pd.testing.assert_frame_equal(first, second)


def test_internal_trade_share_is_a_fraction(two_bloc_graph):
    assignment = detect_trade_blocs(two_bloc_graph)
    assert assignment["internal_trade_share"].between(0.0, 1.0).all()


def test_tightly_clustered_members_trade_mostly_internally(two_bloc_graph):
    assignment = detect_trade_blocs(two_bloc_graph).set_index("country")
    assert assignment.loc["A2", "internal_trade_share"] > 0.9


def test_bloc_summary_covers_every_country(two_bloc_graph):
    assignment = detect_trade_blocs(two_bloc_graph)
    summary = summarize_blocs(assignment)
    assert summary["member_count"].sum() == len(assignment)


def test_inter_bloc_flows_conserve_total_trade(two_bloc_graph):
    assignment = detect_trade_blocs(two_bloc_graph)
    flows = inter_bloc_flows(two_bloc_graph, assignment)
    graph_total = sum(data["weight"] for _, _, data in two_bloc_graph.edges(data=True))
    assert flows["trade_value"].sum() == pytest.approx(graph_total)


# --------------------------------------------------------------------------
# Fragility
# --------------------------------------------------------------------------


@pytest.fixture
def sector_graphs() -> dict:
    """One country brittle in energy, diversified in agriculture."""
    energy = nx.DiGraph()
    energy.add_edge("SoleSupplier", "Importer", weight=100.0)
    energy.add_edge("Tiny", "Importer", weight=1.0)
    energy.add_edge("SoleSupplier", "Other", weight=50.0)

    agriculture = nx.DiGraph()
    for index in range(10):
        agriculture.add_edge(f"Farm{index}", "Importer", weight=10.0)
    agriculture.add_edge("Farm0", "Other", weight=10.0)

    return {"energy": energy, "agriculture": agriculture}


def test_fragility_is_higher_where_supply_is_concentrated(sector_graphs):
    fragility = compute_sector_fragility(sector_graphs, country="Importer").set_index("sector")
    assert fragility.loc["energy", "fragility_score"] > fragility.loc["agriculture", "fragility_score"]
    assert fragility.loc["energy", "supplier_count"] == 2
    assert fragility.loc["agriculture", "supplier_count"] == 10


def test_fragility_names_the_dominant_supplier(sector_graphs):
    fragility = compute_sector_fragility(sector_graphs, country="Importer").set_index("sector")
    assert fragility.loc["energy", "top_supplier"] == "SoleSupplier"
    assert fragility.loc["energy", "top_supplier_share"] > 0.98


def test_sector_ranking_orders_by_supply_concentration(sector_graphs):
    ranking = rank_sector_fragility(sector_graphs)
    assert ranking.iloc[0]["sector"] == "energy"
    assert ranking.iloc[0]["dominant_supplier"] == "SoleSupplier"


def test_fragility_needs_a_comparable_sector():
    with pytest.raises(ValueError, match="at least one sector"):
        compute_sector_fragility({"all": nx.DiGraph()})


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


def test_missing_quality_is_neutral_not_perfect():
    """Absent evidence must not read as perfect evidence."""
    score, reason = data_completeness_score(None)
    assert score == pytest.approx(0.5)
    assert "could not be assessed" in reason


def test_complete_quality_scores_high():
    score, _ = data_completeness_score(
        {"required_field_availability": 1.0, "value_availability": 1.0, "quantity_availability": 1.0}
    )
    assert score == pytest.approx(1.0)


def test_confidence_names_its_weakest_component(simple_graph):
    assessment = build_confidence_assessment(
        data_quality={"required_field_availability": 0.2, "value_availability": 0.2},
        graph=simple_graph,
        country="A",
        propagation_steps=1,
    )
    assert 0.0 <= assessment["score"] <= 1.0
    assert assessment["reason"]
    assert min(assessment["components"], key=assessment["components"].get) == "data_completeness"


def test_more_propagation_steps_lower_confidence(simple_graph):
    shallow = build_confidence_assessment(None, simple_graph, "A", propagation_steps=1)
    deep = build_confidence_assessment(None, simple_graph, "A", propagation_steps=5)
    assert deep["score"] < shallow["score"]


# --------------------------------------------------------------------------
# Country identifier resolution
# --------------------------------------------------------------------------


def test_alias_map_targets_real_iso3_codes():
    """Every alias must point at a code the BACI table actually contains.

    A typo in the alias map would surface as a confusing "not found" for a
    country the module can in fact answer, so this checks the map against the
    real table whenever the cache is present.
    """
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built; alias targets cannot be validated.")

    codes = baci_cache.load_countries(cache_directory())
    known = set(codes["country_iso3"].str.upper())
    unknown = {
        alias: target
        for alias, target in baci_cache.COUNTRY_ALIASES.items()
        if target not in known
    }
    assert not unknown, f"aliases pointing at codes BACI does not have: {unknown}"


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("United States", "USA"),
        ("US", "USA"),
        ("south korea", "Rep. of Korea"),
        ("North Korea", "Dem. People's Rep. of Korea"),
        ("Russia", "Russian Federation"),
        ("Vietnam", "Viet Nam"),
        ("Turkey", "Türkiye"),
        ("Hong Kong", "China, Hong Kong SAR"),
        ("UK", "United Kingdom"),
        ("Czech Republic", "Czechia"),
        ("Burma", "Myanmar"),
    ],
)
def test_common_names_resolve_to_baci_labels(identifier, expected):
    """BACI's formal labels are not what callers type; aliases bridge the gap."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built; country resolution needs the code table.")

    _, resolved = baci_cache.resolve_country(cache_directory(), identifier)
    assert resolved == expected


def test_unknown_country_error_suggests_alternatives():
    """A dead-end error is useless to an orchestrator; suggest a next step."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built; country resolution needs the code table.")

    with pytest.raises(ValueError, match="Did you mean"):
        baci_cache.resolve_country(cache_directory(), "Indai")


def test_iso_numeric_confusion_is_called_out():
    """BACI numeric codes are not ISO 3166 numeric; say so rather than 'not found'."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built; country resolution needs the code table.")

    # 356 is India's ISO 3166 numeric code. India's BACI code is 699.
    with pytest.raises(ValueError, match="own numeric codes"):
        baci_cache.resolve_country(cache_directory(), "356")


def test_year_out_of_range_names_the_coverage():
    from core.data_loader import year_out_of_range_message

    contiguous = year_out_of_range_message(1994, list(range(1995, 2025)))
    assert "1995-2024" in contiguous

    sparse = year_out_of_range_message(2030, [2019, 2020, 2023])
    assert "2019, 2020, 2023" in sparse

    empty = year_out_of_range_message(2000, [])
    assert "build_cache" in empty


def test_mojibake_repair_fixes_double_encoded_names():
    """The CEPII country file ships double-encoded; four names depend on this."""
    from core.data_loader import repair_mojibake

    assert repair_mojibake("TÃ¼rkiye") == "Türkiye"
    assert repair_mojibake("CÃ´te d'Ivoire") == "Côte d'Ivoire"
    assert repair_mojibake("CuraÃ§ao") == "Curaçao"
    assert repair_mojibake("Saint BarthÃ©lemy") == "Saint Barthélemy"


def test_mojibake_repair_leaves_clean_text_alone():
    """Correctly encoded names, and plain ASCII, must pass through untouched."""
    from core.data_loader import repair_mojibake

    for name in ["India", "USA", "China, Hong Kong SAR", "Türkiye", "Curaçao"]:
        assert repair_mojibake(name) == name


def test_cache_country_names_are_not_mojibake():
    """Guards against a cache built before the encoding fix slipping back in."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built.")

    names = baci_cache.load_countries(cache_directory())["country_name"].astype(str)
    # U+00C3 is the tell-tale first character of UTF-8-read-as-Latin-1.
    suspect = [name for name in names if "Ã" in name]
    assert not suspect, f"double-encoded names still in the cache: {suspect}"


# --------------------------------------------------------------------------
# Forecasting on degenerate series
# --------------------------------------------------------------------------


def _series(values):
    """Build a minimal yearly series frame around one metric."""
    years = list(range(2024 - len(values) + 1, 2025))
    return pd.DataFrame({"year": years, "exports": values, "imports": values})


@pytest.mark.parametrize("model", ["auto", "linear", "arima", "hybrid"])
@pytest.mark.parametrize(
    "label, values",
    [
        ("all zeros", [0.0] * 30),
        ("constant", [5.0] * 30),
        ("single point", [7.0]),
        ("two points", [1.0, 2.0]),
        ("ends at zero", [10.0] * 29 + [0.0]),
        ("zero heavy", [0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0] * 3),
        ("steep decline", [1000.0 - (30 * index) for index in range(30)]),
    ],
)
def test_degenerate_series_produce_usable_forecasts(label, values, model):
    """Sparse and flat trade series must not yield NaN, inf, or negatives.

    Real BACI data contains all of these shapes -- micro-territories with a
    handful of non-zero energy years, countries that stop trading a sector
    entirely. A forecast that returns NaN here would propagate straight into
    the API response.
    """
    from core.forecast import forecast_metric_from_series

    result = forecast_metric_from_series(_series(values), "Testland", metric="exports", periods=3, model=model)
    predicted = [point["predicted_value"] for point in result["forecast"]]

    assert len(predicted) == 3
    assert all(np.isfinite(value) for value in predicted), f"{label}/{model}: {predicted}"
    # Trade value cannot be negative, so the metric bound must hold.
    assert all(value >= 0 for value in predicted), f"{label}/{model}: {predicted}"
    assert 0.0 <= result["confidence"] <= 1.0


def test_steep_decline_does_not_forecast_negative_trade():
    """A hard downward trend must clamp at zero, not project negative exports."""
    from core.forecast import forecast_metric_from_series

    values = [1000.0 - (34 * index) for index in range(30)]
    result = forecast_metric_from_series(_series(values), "Testland", metric="exports", periods=3, model="linear")
    assert all(point["predicted_value"] >= 0 for point in result["forecast"])


def test_dependency_ratio_forecasts_stay_within_zero_and_one():
    """Ratios are bounded by definition; the forecast must respect that."""
    from core.forecast import forecast_metric_from_series

    rising = [index / 30.0 for index in range(30)]
    frame = pd.DataFrame({"year": range(1995, 2025), "import_dependency_ratio": rising})
    result = forecast_metric_from_series(
        frame, "Testland", metric="import_dependency_ratio", periods=5, model="linear"
    )
    assert all(0.0 <= point["predicted_value"] <= 1.0 for point in result["forecast"])


def test_forecasting_emits_no_warnings_to_stderr():
    """Non-converging ARIMA fits are handled, so they must not spam the API log."""
    import warnings as warning_module

    from core.forecast import forecast_metric_from_series

    awkward = _series([0.0, 0.0, 4.0, 0.0, 9.0, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0, 0.0, 0.0, 2.0, 0.0])
    with warning_module.catch_warnings(record=True) as captured:
        warning_module.simplefilter("always")
        forecast_metric_from_series(awkward, "Testland", metric="exports", periods=3, model="auto")

    convergence = [w for w in captured if "converge" in str(w.message).lower()]
    assert not convergence, f"convergence warnings leaked: {[str(w.message) for w in convergence]}"


# --------------------------------------------------------------------------
# Cache memo bounds and thread safety
# --------------------------------------------------------------------------


def test_memo_never_exceeds_its_bound():
    """The memo must evict, or a long-running service grows without limit.

    Each entry is a year-sector frame of a few megabytes, so an unbounded
    dict would consume hundreds of megabytes over a sweep of the full
    history and never give it back.
    """
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built.")

    baci_cache.clear_memory()
    cache_dir = cache_directory()
    years = baci_cache.cached_years(cache_dir)

    requested = 0
    for year in years[: baci_cache.MAX_CACHED_YEARS + 6]:
        for sector in ["all", "energy"]:
            baci_cache.load_year(cache_dir, year, sector)
            requested += 1
            assert len(baci_cache._MEMORY) <= baci_cache.MAX_CACHED_YEARS

    assert requested > baci_cache.MAX_CACHED_YEARS, "test did not exceed the bound"
    assert len(baci_cache._MEMORY) == baci_cache.MAX_CACHED_YEARS


def test_memo_evicts_least_recently_used():
    """A key kept warm must survive eviction that drops colder ones."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built.")

    baci_cache.clear_memory()
    cache_dir = cache_directory()
    years = baci_cache.cached_years(cache_dir)
    if len(years) <= baci_cache.MAX_CACHED_YEARS:
        pytest.skip("Not enough cached years to force eviction.")

    warm = years[0]
    baci_cache.load_year(cache_dir, warm, "all")

    for year in years[1 : baci_cache.MAX_CACHED_YEARS]:
        baci_cache.load_year(cache_dir, year, "all")
        baci_cache.load_year(cache_dir, warm, "all")  # keep touching the warm key

    # Fill past the bound; the warm key should still be resident.
    for year in years[baci_cache.MAX_CACHED_YEARS :]:
        baci_cache.load_year(cache_dir, year, "all")
        baci_cache.load_year(cache_dir, warm, "all")

    resident = {key[1] for key in baci_cache._MEMORY}
    assert warm in resident


def test_concurrent_loads_return_identical_data():
    """Threads racing the same key must not observe torn or differing frames."""
    from concurrent.futures import ThreadPoolExecutor

    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built.")

    baci_cache.clear_memory()
    cache_dir = cache_directory()
    year = baci_cache.cached_years(cache_dir)[-1]

    def load(_):
        frame = baci_cache.load_year(cache_dir, year, "all")
        return len(frame), float(frame["trade_value"].sum())

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(load, range(24)))

    assert len(set(results)) == 1, f"threads saw differing data: {set(results)}"


def test_cached_frames_are_isolated_from_callers():
    """A caller mutating its copy must not corrupt the shared memo."""
    from core import baci_cache
    from core.data_loader import cache_directory, cache_ready

    if not cache_ready():
        pytest.skip("No cache built.")

    baci_cache.clear_memory()
    cache_dir = cache_directory()
    year = baci_cache.cached_years(cache_dir)[-1]

    first = baci_cache.load_year(cache_dir, year, "all")
    original_total = float(first["trade_value"].sum())
    first["trade_value"] = 0.0
    first.drop(first.index[:10], inplace=True)

    second = baci_cache.load_year(cache_dir, year, "all")
    assert float(second["trade_value"].sum()) == pytest.approx(original_total)
    assert second.attrs.get("sector") == "all"
