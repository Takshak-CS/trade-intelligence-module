"""The shared response envelope.

Four agents in this system answer different questions over different data, and
an orchestrator has to fan out across all of them and fuse what comes back. It
can only do that if every agent speaks the same shape. Team 128 adopted this
module's envelope as that contract:

    {
      "agent": "trade_intelligence",
      "metadata": { "query_type", "year", "sector", "data_quality", ... },
      "insights": [
        {
          "entity_iso3": "IND",     the shared join key across all four agents
          "entity_name": "India",   human-readable label for the same entity
          "claim":       "...",     one self-contained, readable finding
          "score":       0.42,      sortable; meaning depends on query_type
          "confidence":  0.87,      0-1, how much to trust this claim
          "reason":      "...",     what limited the confidence
          "evidence":    { ... }    the numbers the claim rests on
        }
      ]
    }

Three fields carry most of the integration weight:

``entity_iso3`` is the join key. BACI country names, Gleditsch-Ward numerics,
and FIPS actor codes all resolve to ISO3, which is the only way the fusion
layer can tell that three agents are talking about the same country.

``confidence`` plus ``reason`` is what lets fusion rank and corroborate rather
than blindly concatenate. When two agents disagree about a country, the
orchestrator needs to know which one is standing on firmer ground.

``evidence`` carries the supporting numbers so a fused briefing can cite what a
claim rests on instead of asserting it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# The agent identifier the orchestrator routes on. It must match the name used
# in the shared contract, not the module's internal shorthand.
AGENT_NAME = "trade_intelligence"

# Fields every insight must carry for the fusion layer to consume it.
REQUIRED_INSIGHT_FIELDS = (
    "entity_iso3",
    "entity_name",
    "claim",
    "score",
    "confidence",
    "reason",
    "evidence",
)


def build_insight(
    entity_iso3: Optional[str],
    entity_name: str,
    claim: str,
    score: float,
    confidence: float,
    reason: str,
    evidence: Optional[Mapping[str, object]] = None,
) -> dict:
    """Create a single insight in the shared contract shape.

    ``entity_iso3`` may be None when a row is not a country the crosswalk
    knows. That is reported rather than guessed: a fabricated code would join
    against the wrong entity in another agent's data, which is worse than the
    fusion layer skipping the row.
    """
    return {
        "entity_iso3": str(entity_iso3) if entity_iso3 else None,
        "entity_name": str(entity_name),
        "claim": str(claim),
        "score": float(score),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": str(reason),
        "evidence": dict(evidence or {}),
    }



def build_metadata(
    query_type: str,
    sector: str = "all",
    year: Optional[int] = None,
    country: Optional[str] = None,
    method: Optional[str] = None,
    data_quality: Optional[Mapping[str, object]] = None,
    **extra: object,
) -> dict:
    """Describe what produced a set of insights.

    The orchestrator needs this to align results across agents: two agents
    reporting on different years are not describing the same world, and a
    fusion step that ignores that will produce confident nonsense.
    """
    metadata: dict = {
        "query_type": str(query_type),
        "sector": str(sector),
    }
    if year is not None:
        metadata["year"] = int(year)
    if country:
        metadata["country"] = str(country)
    if method:
        metadata["method"] = str(method)
    metadata["data_quality"] = dict(data_quality) if data_quality else {}

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
