"""Context-efficient web access: summarizing fetch and delegated research."""

from pydantic_ai_harness.web._summarizing_fetch import (
    FetchedPage,
    Fetcher,
    Summarizer,
    SummarizingFetch,
    SummarizingFetchToolset,
)
from pydantic_ai_harness.web._web_research import WebResearch, WebResearchToolset

__all__ = [
    'FetchedPage',
    'Fetcher',
    'SummarizingFetch',
    'SummarizingFetchToolset',
    'Summarizer',
    'WebResearch',
    'WebResearchToolset',
]
