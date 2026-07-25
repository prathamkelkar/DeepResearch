# DeepResearch

A multi-agent research pipeline that takes a topic, autonomously researches it across academic and web sources, and produces a cited, structured report — built to explore agentic system design: bounded reasoning loops, tool-selection judgment, self-correction, and citation integrity.

## What it does

Given a topic, DeepResearch:

1. **Plans** a fixed-then-adaptive research sequence
2. **Researches** the topic using up to six tools — some mandatory, some chosen at runtime based on what's actually missing
3. **Writes** a structured, cited academic-style report from the collected material
4. **Edits** the draft, with the ability to send it back for more research or another writing pass — bounded, so it can't loop forever
5. **Verifies** citations deterministically, rather than trusting the model to reproduce URLs from memory

The output is a Markdown report with inline citations and a References section built entirely from real tool data.

## Architecture

```
Planner → Researcher → Writer ⇄ Editor
              │                    │
     (mandatory: arXiv, Tavily)    │
     (optional: Semantic Scholar,  │
      Wikipedia, web scrape,       │
      code execution)              │
              │                    │
              └── loop back on "needs more research" ──┘
```

**Agents**
- **Planner** — produces a fixed-shape step list (not a free-form plan): mandatory arXiv search, mandatory Tavily search, jugement step(s), drafting, and editing.
- **Researcher** — a single agent reused across all research steps, with the *tools available to it restricted per step* rather than relying on prompting alone. Step 1 gets only arXiv; step 2 gets only Tavily; the judgment step gets the four optional tools (Semantic Scholar, Wikipedia, page scraping, Python execution) and decides which, if any, are actually needed based on what the mandatory searches turned up.
- **Writer** — synthesizes all research into a structured report (abstract, background, methodology, findings, discussion, conclusion), with a word-count floor and automatic expansion retry if the draft comes in short.
- **Editor** — reviews the draft and returns a structured decision (`approved` / `needs_revision` / `needs_more_research`), not free-form prose. This is what drives the self-correction loop.

**Orchestrator** — plain Python control flow, not an LLM call. Routes between agents, enforces loop caps, and handles fallback behavior when something goes wrong (malformed JSON, exhausted retries) so the pipeline always returns *something* usable rather than crashing.

## Design decisions worth knowing

**Tool access is restricted in code, not just in the prompt.** Early versions relied on system-prompt instructions ("only call the tools you're told to"), which the model didn't reliably follow — it would call the same tool multiple times with broadened queries, chasing "better" results. The fix: each research call only receives the tool *definitions* it's actually allowed to use, and once a tool is called once, it's removed from the available set for the rest of that call. This makes over-calling structurally impossible rather than just discouraged.

**Every research call is guaranteed to return real text, never `None`.** If the model exhausts its turn budget mid-tool-call (a real failure mode — models will sometimes keep searching rather than synthesizing), the pipeline forces one final call with no tools available, so the model is mechanically unable to return anything but a text answer.

**Citations are built from data, not memory.** Rather than trusting the writer to reproduce ~30-50 URLs correctly across a long generation (it doesn't — duplicate and occasionally fabricated citations showed up in early testing), every tool result is tagged with a stable ID (`[S1]`, `[S2]`, ...) before the writer sees it. The writer cites by tag; the References section is assembled afterward directly from the original tool data, so a citation can never point to a URL the model invented.

**The revision loop is bounded and routes differently depending on the problem.** The editor distinguishes stylistic issues (`needs_revision` → back to the writer only) from evidence gaps (`needs_more_research` → back to the researcher, then the writer, then the editor again). Each has its own retry cap, so a stubborn editor can't loop the pipeline indefinitely or burn unbounded API calls.

## Tools

| Tool | Role |
|---|---|
| arXiv search | Mandatory — CS/physics/math/stats/econ preprints |
| Tavily search | Mandatory — general web, news, industry sources |
| Semantic Scholar | Optional — peer-reviewed literature outside arXiv's scope |
| Wikipedia | Optional — background/foundational context |
| Web scraper | Optional — full-page depth on promising links, with SSRF protection |
| Python execution | Optional — calculations, verification, quick analysis |

## Model

Runs on NVIDIA's Nemotron 3 family of models via the NVIDIA API.

## Running it

```bash
python main.py
```

Prompts for a topic, runs the full pipeline, writes the final report to `research.md`.

## Known limitations

- Citation *tagging* is fully deterministic; citation *selection* (which sources the writer chooses to use) is still model-driven and not independently fact-checked against claims.
- The optional-tool judgment step is a single reasoning pass — it doesn't currently re-evaluate its own tool choices after seeing results.
- No persistent storage between runs; each topic starts from a clean slate.