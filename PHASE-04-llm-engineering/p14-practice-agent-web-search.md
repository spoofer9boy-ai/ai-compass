# Practice: Agent Web Search

**Phase:** PHASE-04-llm-engineering  
**Subjects Required:** 91 — Agent: Tool Use, 92 — Agent: Planning and Multi-Agent  
**Estimated Time:** 240 minutes  
**Difficulty:** Intermediate

## Industry Context

You are a backend engineer at a market-intelligence startup. Your product team wants an autonomous research agent that can answer ad-hoc questions like "What are the latest funding rounds in climate-tech this quarter?" by planning a sequence of web searches, executing them in parallel, and synthesizing a cited report. The constraint: you cannot rely on a single monolithic LLM call with a long context window—costs scale quadratically and hallucinations rise with token count. Instead, you need a multi-agent system where a planner breaks the query into sub-searches, search agents gather evidence concurrently, and a writer agent distills the results into a markdown report with inline citations.

## The Problem

Build a minimal multi-agent web-search pipeline from first principles. Your system must:

1. **Plan:** Given a user query, generate a structured search plan containing 3–7 independent search queries, each with a stated reason.
2. **Search:** Execute all queries in parallel against a real web-search API, collecting title, URL, and snippet for each result.
3. **Synthesize:** Feed the aggregated search results into a writer agent that produces a markdown report answering the original query, with inline citations linking claims to source URLs.
4. **Trace:** Emit structured logs showing which agent handled which step and how long each step took.

You may use `openai`, `httpx`, and `pydantic`. Do not use LangChain, LlamaIndex, or the OpenAI Agents SDK. The goal is to understand the orchestration glue, not the abstraction.

## Constraints

- Must use a real web-search API. Recommended: Tavily (keyless mode available) or SearXNG (self-hosted, no API key). Do not mock search results.
- Max 10 search calls per user query (rate-limit respect).
- Each search result summary must be ≤ 300 words to keep context size manageable.
- The writer agent must reference source URLs inline using markdown link syntax: `[claim](url)`.
- Total end-to-end latency must stay under 15 seconds for a 5-search plan on a standard residential connection.
- Do not persist state to disk; everything lives in memory for this exercise.

## Starter Code

```python
# starter.py
import asyncio
import time
from typing import List, Optional
from pydantic import BaseModel, Field
import httpx
import os

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
# Use Tavily keyless mode or set TAVILY_API_KEY env var
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
TAVILY_BASE_URL = "https://api.tavily.com"

# Or use a self-hosted SearXNG instance:
# SEARXNG_URL = "http://localhost:8080"

# Your LLM endpoint (OpenAI-compatible)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-fake")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 2. DATA MODELS
# ---------------------------------------------------------------------------
class SearchItem(BaseModel):
    """A single planned search query."""
    query: str
    reason: str

class SearchPlan(BaseModel):
    """The planner's output: a list of searches to perform."""
    searches: List[SearchItem] = Field(..., min_length=1, max_length=10)

class SearchResult(BaseModel):
    """Raw result from one search query."""
    query: str
    title: str
    url: str
    snippet: str

class Report(BaseModel):
    """The writer's final output."""
    short_summary: str
    markdown_report: str
    sources: List[str]

# ---------------------------------------------------------------------------
# 3. LLM CLIENT
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
    ) -> str:
        """Call the LLM and return the raw text content."""
        # TODO: Implement the HTTP POST to /chat/completions
        pass

# ---------------------------------------------------------------------------
# 4. SEARCH CLIENT
# ---------------------------------------------------------------------------
class SearchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str) -> List[SearchResult]:
        """Execute a single web search and return parsed results."""
        # TODO: Call Tavily /search endpoint (keyless or authenticated)
        # or call SearXNG /search?q=...&format=json
        # Return a list of SearchResult objects.
        pass

# ---------------------------------------------------------------------------
# 5. AGENTS
# ---------------------------------------------------------------------------
class PlannerAgent:
    """Breaks a user query into a structured SearchPlan."""

    SYSTEM_PROMPT = (
        "You are a research planner. Given a user question, produce a JSON object with a "
        "'searches' array. Each item must have 'query' (the search string) and 'reason' "
        "(why this search helps answer the question). Output 3 to 7 searches."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def plan(self, query: str) -> SearchPlan:
        # TODO: Call LLM with SYSTEM_PROMPT + user query.
        # Parse the JSON response into a SearchPlan.
        pass

class SearchAgent:
    """Executes one search and returns a concise summary."""

    SYSTEM_PROMPT = (
        "You are a research assistant. Given search results, produce a concise summary "
        "(≤ 300 words) capturing the main points relevant to the original query. "
        "Write succinctly; incomplete sentences are fine. Do not add commentary beyond the summary."
    )

    def __init__(self, llm: LLMClient, search: SearchClient):
        self.llm = llm
        self.search = search

    async def run(self, item: SearchItem) -> str:
        # TODO: 1) Call SearchClient.search(item.query)
        #       2) Feed results into LLM with SYSTEM_PROMPT
        #       3) Return the summary string
        pass

class WriterAgent:
    """Synthesizes search summaries into a final markdown report."""

    SYSTEM_PROMPT = (
        "You are a senior analyst. Given an original question and a set of search summaries, "
        "write a markdown report that directly answers the question. Cite sources inline using "
        "[claim](url) syntax. Include a short_summary (2–3 sentences) and a sources list."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def write(self, query: str, summaries: List[str], results: List[SearchResult]) -> Report:
        # TODO: Build a prompt containing the query, summaries, and source URLs.
        #       Call the LLM and parse the response into a Report.
        pass

# ---------------------------------------------------------------------------
# 6. ORCHESTRATOR
# ---------------------------------------------------------------------------
class ResearchOrchestrator:
    def __init__(self, planner: PlannerAgent, search_agent: SearchAgent, writer: WriterAgent):
        self.planner = planner
        self.search_agent = search_agent
        self.writer = writer

    async def run(self, query: str) -> Report:
        start = time.perf_counter()

        # Step 1: Plan
        plan = await self.planner.plan(query)
        print(f"[Planner] {len(plan.searches)} searches planned in {time.perf_counter() - start:.2f}s")

        # Step 2: Execute searches in parallel
        # TODO: Create asyncio.gather or asyncio.TaskGroup for all search items

        # Step 3: Write report
        # TODO: Pass summaries + raw results to writer

        total = time.perf_counter() - start
        print(f"[Orchestrator] Total time: {total:.2f}s")
        return report

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
async def main():
    llm = LLMClient(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL)
    search_client = SearchClient(api_key=TAVILY_API_KEY)

    planner = PlannerAgent(llm)
    search_agent = SearchAgent(llm, search_client)
    writer = WriterAgent(llm)

    orchestrator = ResearchOrchestrator(planner, search_agent, writer)

    query = input("Research question: ").strip()
    report = await orchestrator.run(query)

    print("\n===== REPORT =====\n")
    print(report.markdown_report)
    print("\n===== SOURCES =====\n")
    for src in report.sources:
        print(f"- {src}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Evaluation Criteria

1. **Planning quality:** The planner outputs a valid `SearchPlan` with 3–7 diverse, relevant queries. No duplicate or empty queries.
2. **Search execution:** All planned searches run concurrently (not sequentially). Each returns at least one `SearchResult` with a real URL.
3. **Summarization:** Each search summary is ≤ 300 words and captures the essence of the results—not a raw dump of snippets.
4. **Citation fidelity:** The writer's report contains inline markdown citations `[claim](url)` that map to real URLs returned by the search client. No fabricated URLs.
5. **Latency:** A 5-search plan completes end-to-end in under 15 seconds on a standard connection (excluding model download time).
6. **Error handling:** If a single search fails (network error, rate limit), the orchestrator continues with the remaining searches and notes the failure in the final report.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
# solution.py
import asyncio
import json
import time
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import httpx
import os

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
TAVILY_BASE_URL = "https://api.tavily.com"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-fake")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 2. DATA MODELS
# ---------------------------------------------------------------------------
class SearchItem(BaseModel):
    query: str
    reason: str

class SearchPlan(BaseModel):
    searches: List[SearchItem] = Field(..., min_length=1, max_length=10)

class SearchResult(BaseModel):
    query: str
    title: str
    url: str
    snippet: str

class Report(BaseModel):
    short_summary: str
    markdown_report: str
    sources: List[str]

# ---------------------------------------------------------------------------
# 3. LLM CLIENT
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# 4. SEARCH CLIENT (Tavily keyless or authenticated)
# ---------------------------------------------------------------------------
class SearchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(self, query: str) -> List[SearchResult]:
        payload = {
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": 5,
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        resp = await self.client.post(
            f"{TAVILY_BASE_URL}/search",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    query=query,
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
            )
        return results

# ---------------------------------------------------------------------------
# 5. AGENTS
# ---------------------------------------------------------------------------
class PlannerAgent:
    SYSTEM_PROMPT = (
        "You are a research planner. Given a user question, produce a JSON object with a "
        "'searches' array. Each item must have 'query' (the search string) and 'reason' "
        "(why this search helps answer the question). Output 3 to 7 searches. "
        "Respond ONLY with valid JSON."
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def plan(self, query: str) -> SearchPlan:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ]
        raw = await self.llm.chat_completion(messages, temperature=0.3)
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
        cleaned = cleaned.strip()
        return SearchPlan.model_validate_json(cleaned)

class SearchAgent:
    SYSTEM_PROMPT = (
        "You are a research assistant. Given search results, produce a concise summary "
        "(≤ 300 words) capturing the main points relevant to the original query. "
        "Write succinctly; incomplete sentences are fine. Do not add commentary beyond the summary."
    )

    def __init__(self, llm: LLMClient, search: SearchClient):
        self.llm = llm
        self.search = search

    async def run(self, item: SearchItem) -> str:
        results = await self.search.search(item.query)
        if not results:
            return f"No results for query: {item.query}"

        context = "\n\n".join(
            f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
            for r in results
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original query: {item.query}\n\nSearch results:\n{context}",
            },
        ]
        summary = await self.llm.chat_completion(messages, temperature=0.3)
        return summary

class WriterAgent:
    SYSTEM_PROMPT = (
        "You are a senior analyst. Given an original question and a set of search summaries, "
        "write a markdown report that directly answers the question. Cite sources inline using "
        "[claim](url) syntax. Include a short_summary (2–3 sentences) and a sources list. "
        "Respond ONLY with valid JSON matching the Report schema: "
        '{"short_summary": "...", "markdown_report": "...", "sources": ["url1", "url2"]}'
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def write(self, query: str, summaries: List[str], results: List[SearchResult]) -> Report:
        context = "\n\n---\n\n".join(summaries)
        source_map = "\n".join(f"- {r.url}: {r.title}" for r in results)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Search summaries:\n{context}\n\n"
                    f"Available sources:\n{source_map}"
                ),
            },
        ]
        raw = await self.llm.chat_completion(
            messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
        cleaned = cleaned.strip()
        return Report.model_validate_json(cleaned)

# ---------------------------------------------------------------------------
# 6. ORCHESTRATOR
# ---------------------------------------------------------------------------
class ResearchOrchestrator:
    def __init__(self, planner: PlannerAgent, search_agent: SearchAgent, writer: WriterAgent):
        self.planner = planner
        self.search_agent = search_agent
        self.writer = writer

    async def run(self, query: str) -> Report:
        start = time.perf_counter()

        plan = await self.planner.plan(query)
        plan_time = time.perf_counter() - start
        print(f"[Planner] {len(plan.searches)} searches planned in {plan_time:.2f}s")

        search_start = time.perf_counter()
        tasks = [asyncio.create_task(self.search_agent.run(item)) for item in plan.searches]
        summaries = []
        for task in asyncio.as_completed(tasks):
            try:
                summary = await task
                summaries.append(summary)
            except Exception as exc:
                summaries.append(f"Search failed: {exc}")
        search_time = time.perf_counter() - search_start
        print(f"[Search] {len(summaries)} searches completed in {search_time:.2f}s")

        # Re-run searches to collect raw results for citation (or cache them in SearchAgent)
        # For simplicity, re-run here; in production, cache results in SearchAgent.
        raw_results: List[SearchResult] = []
        for item in plan.searches:
            try:
                raw_results.extend(await self.search_agent.search.search(item.query))
            except Exception:
                pass

        write_start = time.perf_counter()
        report = await self.writer.write(query, summaries, raw_results)
        write_time = time.perf_counter() - write_start
        print(f"[Writer] Report written in {write_time:.2f}s")

        total = time.perf_counter() - start
        print(f"[Orchestrator] Total time: {total:.2f}s")
        return report

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
async def main():
    llm = LLMClient(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=LLM_MODEL)
    search_client = SearchClient(api_key=TAVILY_API_KEY)

    planner = PlannerAgent(llm)
    search_agent = SearchAgent(llm, search_client)
    writer = WriterAgent(llm)

    orchestrator = ResearchOrchestrator(planner, search_agent, writer)

    query = input("Research question: ").strip()
    report = await orchestrator.run(query)

    print("\n===== REPORT =====\n")
    print(report.markdown_report)
    print("\n===== SOURCES =====\n")
    for src in report.sources:
        print(f"- {src}")

if __name__ == "__main__":
    asyncio.run(main())
```

</details>

## What You Actually Learned

- **Agent: Tool Use:** You built a `SearchClient` tool that the LLM does not call directly; instead, you orchestrated the tool invocation explicitly in Python. This separates the "what" (reasoning) from the "how" (API calls), giving you full control over retries, timeouts, and rate limits.
- **Agent: Planning and Multi-Agent:** You decomposed a vague user request into a structured plan, then fanned out independent search tasks across concurrent agents. The planner, search agent, and writer agent each had a single responsibility—mirroring how real production systems avoid monolithic prompts.
- **Async orchestration:** You used `asyncio.gather` (or `asyncio.as_completed`) to parallelize I/O-bound search calls, cutting latency from sequential seconds to the slowest single call.
- **Citation hygiene:** You enforced inline markdown citations tied to real URLs, preventing the common failure mode where LLMs hallucinate sources. The writer only had access to URLs that the search client actually returned.

## Appendix

### Common Pitfalls

- **Blocking the event loop:** Do not use `requests` inside async agents; it blocks the entire orchestrator. Always use `httpx.AsyncClient` or `aiohttp`.
- **Context explosion:** Feeding raw HTML into the LLM wastes tokens. Summarize search snippets first, then pass summaries to the writer.
- **Fabricated URLs:** If the writer invents URLs, tighten the prompt by including an explicit "Available sources" list and instructing the model to only cite from that list.

### Variations to Try

1. **SearXNG backend:** Replace Tavily with a local SearXNG instance for fully private, self-hosted search. The API shape is similar (`/search?q=...&format=json`).
2. **Streaming writer:** Instead of waiting for all searches to finish, stream partial summaries to the writer as they arrive using `asyncio.Queue`.
3. **Human-in-the-loop:** After the planner generates a search plan, pause and ask the user to approve, edit, or reject individual search queries before execution.

### Further Reading

- [OpenAI Agents SDK: Research Bot](https://github.com/openai/openai-agents-python/tree/main/examples/research_bot) — Production-grade multi-agent research pipeline from OpenAI.
- [Tavily Python SDK](https://github.com/tavily-ai/tavily-python) — Keyless web-search API with structured output.
- [SearXNG Documentation](https://docs.searxng.org) — Self-hosted metasearch engine; no API keys required.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — The paper that formalized interleaving thought and tool use.
- [CS294 LLM Agents Notes (UC Berkeley)](https://github.com/rajdeepmondaldotcom/CS294_LLM_Agents_Notes_Fall2024) — Lecture notes on reasoning, tool use, and multi-agent frameworks.
