"""Structured output helpers for the trade agent.

Every query type in this module returns the same envelope, because the trade
agent is one of four agents whose output an orchestrator has to fuse. A
consistent shape means the orchestrator can rank, filter, and merge insights
across agents without knowing which one produced them.

    {
      "agent": "trade",
      "metadata": {...},          what was asked, and what answered it
      "insights": [               ranked findings, most significant first
        {
          "country": "India",
          "score": 0.42,          meaning depends on query_type, always sortable
          "summary": "...",       one self-contained sentence a human can read
          "confidence": 0.87,     how much to trust this insight
          ...                     query-specific fields
        }
      ]
    }

The confidence field is the part that matters most for fusion. When two agents
disagree about a country, the orchestrator needs to know which one is standing
on firmer ground, and a bare score cannot express that.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

AGENT_NAME = "trade"


def build_insight(country: str, score: float, summary: str, confidence: float) -> dict:
    """Create a single standardized insight record."""
    return {
        "country": str(country),
        "score": float(score),
        "summary": str(summary),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }



def build_metadata(
    query_type: str,
    sector: str = "all",
    year: Optional[int] = None,
    country: Optional[str] = None,
    method: Optional[str] = None,
    **extra: object,
) -> dict:
    """Describe what produced a set of insights.

    The orchestrator needs this to align results across agents: two agents
    reporting on different years are not describing the same world, and a
    fusion step that ignores that will produce confident nonsense.
    """
    metadata = {
        "query_type": str(query_type),
        "sector": str(sector),
    }
    if year is not None:
        metadata["year"] = int(year)
    if country:
        metadata["country"] = str(country)
    if method:
        metadata["method"] = str(method)

    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata



def format_agent_output(
    insights: Iterable[Mapping[str, object]],
    agent_name: str = AGENT_NAME,
    metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """Wrap a list of insights in the agent response format."""
    payload = {
        "agent": agent_name,
        "insights": [dict(insight) for insight in insights],
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return payload
