"""PydanticAI Agent prototype for structured research tasks."""

from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import KnownModelName, Model

from app.schemas.research import (
    ResearchResult,
    ResearchSource,
    SourceType,
)


@dataclass
class MockSearchClient:
    """Mock search client simulating external data retrieval for tools."""

    sources_database: list[ResearchSource] = field(
        default_factory=lambda: [
            ResearchSource(
                url="https://langchain-ai.github.io/langgraph/",
                title="LangGraph Documentation - Workflow Orchestration",
                source_type=SourceType.DOCUMENTATION,
                reliability_score=0.95,
            ),
            ResearchSource(
                url="https://ai.pydantic.dev/",
                title="PydanticAI - Production Agentic Type Safety",
                source_type=SourceType.DOCUMENTATION,
                reliability_score=0.98,
            ),
            ResearchSource(
                url="https://dspy-docs.vercel.app/",
                title="DSPy Declarative Language Model Optimization",
                source_type=SourceType.DOCUMENTATION,
                reliability_score=0.92,
            ),
        ]
    )

    async def search(self, query: str, limit: int = 5) -> list[ResearchSource]:
        """Simulate async search filtering."""
        query_terms = query.lower().split()
        matched = [
            src
            for src in self.sources_database
            if any(term in src.title.lower() for term in query_terms)
        ]
        return matched[:limit] if matched else self.sources_database[:limit]


@dataclass
class ResearchDependencies:
    """Dependency container injected into PydanticAI Agent runs."""

    search_client: MockSearchClient = field(default_factory=MockSearchClient)
    min_confidence: float = 0.6
    max_sources: int = 3


def create_research_agent(
    model: str | Model | KnownModelName = "test",
) -> Agent[ResearchDependencies, ResearchResult]:
    """Factory creating a PydanticAI Agent configured for structured ResearchResult output."""
    agent: Agent[ResearchDependencies, ResearchResult] = Agent(
        model,
        deps_type=ResearchDependencies,
        output_type=ResearchResult,
        system_prompt=(
            "You are a technical research agent. "
            "Use available tools to fetch research sources before generating the final report. "
            "Return a strictly structured ResearchResult matching the required schema."
        ),
    )

    @agent.tool
    async def fetch_research_sources(
        ctx: RunContext[ResearchDependencies], query: str
    ) -> list[ResearchSource]:
        """Search and retrieve verified technical research sources."""
        return await ctx.deps.search_client.search(query=query, limit=ctx.deps.max_sources)

    @agent.output_validator
    async def validate_research_result(
        ctx: RunContext[ResearchDependencies], result: ResearchResult
    ) -> ResearchResult:
        """Enforce domain rules and trigger automatic agent retries when confidence is too low."""
        if result.summary.confidence < ctx.deps.min_confidence:
            raise ModelRetry(
                f"Confidence score {result.summary.confidence:.2f} is below the required "
                f"minimum threshold of {ctx.deps.min_confidence:.2f}. "
                "Please re-evaluate the source material and refine key findings."
            )
        return result

    return agent
