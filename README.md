# Co-Scientist Multi-Agent Framework — JARVIS Edition

**An autonomous agent that executes any natural-language task — with the Co-Scientist hypothesis pipeline preserved as a built-in research tool.**

Co-Scientist has evolved from a hypothesis-generation pipeline into a general-purpose autonomous agent ("JARVIS-in-a-box"): give it a task in plain English and it **plans, acts in a sandbox, observes, and verifies** — looping until the work is actually done, not until it has produced text about the work. The original Co-Scientist research pipeline (generation → reflection → evolution → ranking → meta-review) is **kept intact** and invoked automatically for scientific research questions.

```bash
co-scientist task "compute the 50th Fibonacci number and write it to fib.txt"
co-scientist task "research novel drug targets for Alzheimer's disease"   # → routes to the research pipeline
co-scientist task --graph "research X, write a summary, and create slides" # → parallel task DAG
```

## JARVIS Mode — how it works

| Capability | What it does | Where |
|---|---|---|
| **5-stage worker loop** | Think → Plan → Act → Observe → **Verify**. Terminates only on a *verified* `FINAL_ANSWER`, max-turns, or budget — never on "produced some text". | `agent/worker.py` |
| **Execution sandbox** | Runs real shell commands. Provider resolves `e2b → docker → local` automatically, so it works on a laptop and scales to a cloud VM. | `infrastructure/execution_sandbox.py` |
| **Task graph (DAG)** | Decomposes a goal into a dependency graph and runs ready nodes in parallel; one node failing does not kill independent branches. | `task_graph/`, `agent/command_center.py` |
| **Dynamic tool forge** | No tool for the job? Vector-search the library; if absent, write a script, test it in the sandbox, and register it for reuse. | `agent/tools/tool_forge.py` |
| **Guardian + file-level audit** | Every command/file write is risk-classified and gated; destructive actions (`rm -rf`, `mkfs`, …) are refused without permission. Every action is logged for replay/rollback. | `governance/` |
| **Tools** | `bash` (primary), file ops, web search/fetch (prompt-injection-guarded), MCP client, computer-use. | `agent/tools/` |

> **Note on `17! mod 1000`:** CLAUDE.md's example expects `880`, but `17!` has three trailing zeros, so the mathematically correct result is **`0`** — and the agent computes and verifies the true value rather than echoing the spec's literal.

---

## Architecture

```text
User → CLI / TUI
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     SwarmOrchestrator                         │
│  (K2.6 plans parallel sub‑agent tasks)                        │
│  ┌────────────┬────────────┬────────────┬──────────────┐     │
│  │ Generation │   Search   │ Reflection │  Evolution   │     │
│  │  workers   │  workers   │  workers   │   workers    │     │
│  └────────────┴────────────┴────────────┴──────────────┘     │
│                     │ (asyncio.gather)                        │
│                     ▼                                         │
│               Merge Gate & GovernanceGate                     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
   Hypothesis Pool (ChromaDB + SQLite)
        │
        ▼
   Ranking (TrueSkill / ELO tournament)
        │
        ▼
   Meta‑Review & Self‑Improvement Loop
        │
        ▼
   Final Leaderboard + Recommendations

### Core Agents (Co-Scientist Pipeline)

```

| Agent | Role |
|-------|------|
| **Supervisor** | Top-level orchestrator. Parses goals, configures pipeline, manages workers. |
| **Generation** | Brainstorms ideas, searches literature, generates novel climate hypotheses grounded in physical reality. |
| **Reflection** | Critical reviewer. Fact-checks, evaluates novelty, identifies flaws, and runs **3-Layer Safety checks**. |
| **Proximity** | Maps hypotheses to semantic vector space (HDBSCAN) to detect redundant concepts. |
| **Evolution** | Refines hypotheses using LLM-driven critique integration, combines ideas, bridges logical gaps. |
| **Ranking** | ELO tournament system. Head-to-head debates judged by LLM weighing Climate Impact, Safety, and Equity. |
| **Meta-review** | Cross-agent evaluation, final quality assessment. |

### Robin Closed-Loop Agents (Climate Validation)

| Agent | Role |
|-------|------|
| **Osprey** | Climate literature search (EarthArXiv, IPCC reports, Semantic Scholar). Updates Epistemic Uncertainty. |
| **Condor** | Deep analysis. Evaluates scenario alignment, carbon budget viability, and tipping point proximity. |
| **Albatross** | Data execution. Integrates with the FaIR climate emulator and runs Docker-sandboxed Python code on climate datasets (e.g. `xarray` over Copernicus data) with an **8-instance consensus**. |

### Key Innovation: Dynamic ELO Ranking & Epistemic Uncertainty

The Ranking Agent uses a modified TrueSkill rating system to evaluate hypotheses:
1. **Head-to-head Debates**: Structured LLM evaluation judging Climate Impact, Safety (Tipping Points), and Equity.
2. **Epistemic Uncertainty Tracking**: Decoupled from TrueSkill `sigma`. Contradictory literature or analysis disagreement widens uncertainty; consensus narrows it.
3. **Combined Leaderboard**: Ranks hypotheses based on `conservative_trueskill - (epistemic_uncertainty * 10)`. Flags highly-ranked but highly-uncertain hypotheses for human governance review.

### Key Innovation: Climate Data & Emulator Integration

The system natively bridges to real-world climate tooling:
- **FaIR Emulator**: Built-in wrapper for the widely-used FaIR (Finite Amplitude Impulse Response) simple climate model, projecting temperatures across SSP scenarios.
- **Copernicus CDS (Async)**: Integrates with the Copernicus Climate Data Store using non-blocking background polling threads so long-running planetary data downloads do not block agent pipelines.

### Key Innovation: 3-Layer Climate Safety Guardrails

A strict `ClimateSafetyAssessor` scrutinizes every intervention:
1. **Layer 1: Tipping Points**: Rejects interventions severely risking AMOC collapse, Greenland Ice Sheet melt, etc.
2. **Layer 2: Planetary Boundaries**: Quarantines interventions threatening biosphere integrity or ocean acidification.
3. **Layer 3: Precautionary Principle**: SRM (Solar Radiation Modification) interventions mandate hard governance gates due to termination shock risks.

### Key Innovation: Secure Docker Sandbox execution

Data extraction tools generated by the AI (like filtering petabytes of ERA5 global climate data with `xarray`) run in a strict **2GB memory-limited Docker sandbox** with no network access. A static pre-execution validator blocks infinite `.compute()` RAM spikes on unbounded datasets, requiring bounding box selections `.sel()` first.

## Installation

### Using UV (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

uv pip install -e ".[all]"
```

### Using pip

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Initialize Project

```bash
co-scientist init --path my_research
cd my_research
cp .env.example .env
# Edit .env with your LLM, Search, and Climate API keys
```

### 2. Run an autonomous task (JARVIS)

```bash
# Any natural-language task — runs the 5-stage worker loop in a sandbox
co-scientist task "compute 2^64 and write it to power.txt"

# Decompose a multi-part goal into a parallel task DAG
co-scientist task --graph "summarise the latest on RAG, then draft 3 slide titles"

# Force the worker loop (skip intent routing) or force research routing
co-scientist task --worker "list the 5 largest files under /etc by size"
co-scientist task --research "novel catalysts for green hydrogen"

# Inspect what an agent did (file-level audit replay / crash recovery)
co-scientist resume --limit 50
```

> Requires an LLM: set `OPENROUTER_API_KEY` (free Kimi K2.6 default) in `.env`, or point `llm.provider`/`fallback_chain` at a local Ollama for zero-key use. Docker is used for sandboxing when available; otherwise it falls back to a project-scoped local subprocess.

### 3. Run Research

```bash
# Unlimited mode — runs until convergence or manual stop
co-scientist discover "Find novel marine carbon removal with TRL > 6" --converge

# With safety cap (max 20 cycles)
co-scientist discover "Evaluate stratospheric aerosol injection risks" --max-cycles 20

# Old capped mode (basic research cycle)
co-scientist research "Find novel Solar Radiation Modification (SRM) methods"

# With closed-loop validation (Robin)
co-scientist research "Ocean Alkalinity Enhancement deployment strategies" --closed-loop --iterations 3

# Full options
co-scientist research "Novel Carbon Dioxide Removal via enhanced rock weathering" \
    --domain climate_science \
    --iterations 5 \
    --closed-loop \
    --output results.json
```

### 4. Check Status

```bash
# View leaderboard
co-scientist leaderboard --top 10

# List agents
co-scientist agents
```

## Configuration

Create a `config.yaml` file:

```yaml
project_name: climate-intervention-discovery
max_iterations: 10

llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7

climate:
  ssp_scenarios: ["SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
  fair_version: "2.1"
  use_fair_for_screening: true

agents:
  max_hypotheses_per_batch: 10
  similarity_threshold: 0.85
  min_debates_per_hypothesis: 10
  albatross_consensus_instances: 8
  albatross_consensus_tolerance: 0.1
  docker_memory_limit: "2g"

database:
  vector_db_path: "./data/vector_db"
  checkpoint_dir: "./data/checkpoints"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `SEARCH__SERPER_API_KEY` | Serper.dev API key for Google search |
| `SEARCH__TINYFISH_API_KEY` | Tinyfish API key for markdown extraction |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key |
| `CLIMATE__CDS_API_KEY` | Copernicus API Key |
| `CLIMATE__EARTHDATA_USERNAME` | NASA EarthData Username |
| `CLIMATE__EARTHDATA_PASSWORD` | NASA EarthData Password |

## Testing

### Automated Test Suite
```bash
# Run automated tests
PYTHONPATH=src pytest tests/ -v
```

## License

MIT License - see LICENSE file for details.
