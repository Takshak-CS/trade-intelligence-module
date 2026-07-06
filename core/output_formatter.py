"""Structured output helpers for the trade agent."""

from __future__ import annotations

from typing import Iterable, Mapping


def build_insight(country: str, score: float, summary: str, confidence: float) -> dict:
    """Create a single standardized insight record."""
    return {
        "country": str(country),
        "score": float(score),
        "summary": str(summary),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }



def format_agent_output(insights: Iterable[Mapping[str, object]], agent_name: str = "trade") -> dict:
    """Wrap a list of insights in the agent response format."""
    return {"agent": agent_name, "insights": [dict(insight) for insight in insights]}
