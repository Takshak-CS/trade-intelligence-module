"""Feature engineering for country-level trade metrics."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd


def compute_total_imports(graph: nx.DiGraph) -> Dict[str, float]:
    """Compute total imports for each country in the graph."""
    imports: Dict[str, float] = {}
    for country in graph.nodes:
        imports[country] = float(
            sum(edge_data.get("weight", 0.0) for _, _, edge_data in graph.in_edges(country, data=True))
        )
    return imports


def compute_total_exports(graph: nx.DiGraph) -> Dict[str, float]:
    """Compute total exports for each country in the graph."""
    exports: Dict[str, float] = {}
    for country in graph.nodes:
        exports[country] = float(
            sum(edge_data.get("weight", 0.0) for _, _, edge_data in graph.out_edges(country, data=True))
        )
    return exports


def compute_country_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Compute country-level import, export, and dependency features."""
    imports = compute_total_imports(graph)
    exports = compute_total_exports(graph)

    records = []
    for country in sorted(graph.nodes):
        total_imports = imports.get(country, 0.0)
        total_exports = exports.get(country, 0.0)
        trade_total = total_imports + total_exports
        import_dependency_ratio = total_imports / trade_total if trade_total else 0.0
        export_dependency_ratio = total_exports / trade_total if trade_total else 0.0

        records.append(
            {
                "country": country,
                "total_imports": float(total_imports),
                "total_exports": float(total_exports),
                "trade_total": float(trade_total),
                "import_dependency_ratio": float(import_dependency_ratio),
                "export_dependency_ratio": float(export_dependency_ratio),
            }
        )

    return pd.DataFrame(records)
