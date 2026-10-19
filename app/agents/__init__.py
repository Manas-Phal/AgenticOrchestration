"""Agents package for PydanticAI agent definitions."""

from app.agents.research_agent import (
    MockSearchClient,
    ResearchDependencies,
    create_research_agent,
)

__all__ = [
    "ResearchDependencies",
    "MockSearchClient",
    "create_research_agent",
]
