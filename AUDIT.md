# JARVIS Migration Audit — Before / After

> Scope: transform the product per `specs/CLAUDE (2).md` (the behavioral contract) into an
> industry-ready, usable autonomous agent, then audit what changed. This document records the
> **before** state, **what was done** (phase by phase, with verification), and every place the
> spec diverged from reality and how it was resolved.
> Date: 2026-06-14.

---

## 1. Executive summary

The repo was a Co-Scientist hypothesis-generation pipeline (~5,000 LOC) that **did not install,
did not import, and had zero runnable tests**. It has been transformed into the spec's
"JARVIS-in-a-box": a general-purpose autonomous agent driven by a `co-scientist task "..."`
command, while the original research pipeline is **preserved intact** as a built-in sub-tool.

| | Before | After |
|---|---|---|
| Installable (`pip install -e .`) | ❌ (broken `pyproject`, no `src` layout) | ✅ |
| Package imports | ❌ (`numpy`/`trueskill`/`structlog` missing; deps not declared as installed) | ✅ |
| Runnable tests | **0** (only test file imported a deleted module) | **79 passing**, 1 skipped |
| `task` command / worker loop | ❌ none | ✅ 5-stage Think→Plan→Act→Observe→Verify |
| Verify stage | ❌ absent (root cause of "50 hypotheses, 0 deliverables") | ✅ deterministic shell-check + LLM-judge |
| Task DAG / parallel decomposition | ❌ | ✅ fault-isolated scheduler |
| Dynamic tool forge | ❌ | ✅ vector-lookup → forge → test → register → reuse |
| Guardian + file-level audit | partial (`GovernanceGate`, hypothesis-only) | ✅ gates every command/file-write, replay/rollback |
| Sandbox | Python-only `DockerSandbox`, errored without Docker | ✅ `ExecutionSandbox` runs **shell**, e2b→docker→local fallback |
| Research pipeline | the whole product | preserved + still imports (sub-tool) |
| New code | — | ~3,000 LOC implementation + ~650 LOC tests |

---

## 2. The "before" state (audited, not assumed)

A parallel read of every subsystem established the actual starting point:

- **Build was broken.** `pyproject.toml` had a duplicated `[project.optional-dependencies] dev`
  key (a TOML "cannot overwrite a value" parse error that broke `pytest` config loading) **and**
  no `[tool.setuptools.packages.find]`, so the `src/`-layout package was never discoverable by an
  editable install.
- **Tests were dead.** `tests/mock_llm.py` imported `BaseLLMProvider, Message, LLMResponse` from
  `infrastructure/llm.py` — **none of those symbols exist** there (it defines `OpenAIProvider`,
  `LLMRouter`, …). The only real test module, `tests/test_climate_models.py`, imported
  `co_scientist.core.climate_models`, a module **not present** in the tree. Net: collection failed,
  **zero tests ran**.
- **Dependencies were not installed** (`numpy`, `trueskill`, `structlog`, `cachetools` …), so
  `import co_scientist` raised immediately.
- The pipeline had **no Verify stage** in `orchestrator._run_cycle`, exactly as the spec's §1.1
  diagnosis states.

### 2.1 The §5 "known bugs" — actual status vs. the spec

The spec lists 5 bugs to "fix before adding features." Audited against the real code:

| Bug | Spec claim | Reality | Action |
|---|---|---|---|
| **#1** `ConvergenceTracker.is_converged` index error | crashes at window boundary | **already fixed** — `task_manager.py` already had the spec's "CORRECT" `tail_start` guard | locked with a regression test |
| **#2** `get_flagged()` returns everything | filter broken | **already fixed** — `_FLAGGED_STATUSES` set filter present | locked with a regression test |
| **#3** `get_leaderboard()` missing `adjusted_score` | key absent | **already fixed** — key present in both `to_display()` and `HypothesisPool.get_leaderboard()` | locked with a regression test |
| **#4** TUI `_refresh_leaderboard()` wrong keys | in `cli/tui.py` | **file does not exist** — no `tui.py` anywhere; the real `leaderboard` command already reads the correct keys | non-applicable (recorded) |
| **#5** validation stubs return mocks | always `{"status":"validated","evidence":[]}` | **real** — all three validators were unconditional stubs | **fixed** (see §3 Phase 0) |

Per §11.1 ("surface, don't hide") and §11.3 ("don't "fix" code that isn't broken"), bugs 1–3 were
**not re-edited**; instead regression tests were added that would fail against the spec's "WRONG"
code, locking the existing fixes.

---

## 3. What was done — phase by phase (each with its verification)

The spec's own §7 phase order and §11.4 ("don't advance until the Done-when passes") were followed,
with the §12 rule of dispatching parallel sub-agents for independent work.

### Phase 0 — Foundation
- Fixed `pyproject.toml` (removed the duplicate `dev` key; added `src`-layout package discovery;
  added a `[tool.pytest.ini_options]` block with `asyncio_mode=auto`, `pythonpath=["src"]`).
- Repaired `tests/mock_llm.py` → a real `MockLLMRouter` matching the actual
  `LLMRouter.complete(messages, response_format)` interface.
- Guarded the dead legacy test module with `pytest.importorskip` so the suite collects green
  (no deletion of pre-existing code).
- Implemented **bug #5**: `LiteratureValidator`, `SimulationValidator`, `ConsensusValidator` now
  inspect the hypothesis and can return `REJECTED`/`UNDER_REVIEW`/`CONSENSUS_PASSED` (LLM-backed
  with a deterministic heuristic fallback).
- Added the §6 config sections (`agent`, `sandbox`, `tool_forge`, `swarm`) + updated
  `config.yaml` / `.env.example`; wired `OPENROUTER_API_KEY` through to the OpenRouter provider.
- **Verify:** installs, imports, `pytest` green.

### Phase 1 — `task` command
- `co-scientist task "..."` with an `IntentParser` that routes research/hypothesis requests to the
  preserved pipeline and everything else to the worker loop. `research`/`discover` unchanged.
- **Verify:** command registered; intent routing unit-tested; graceful "no LLM configured" message.

### Phase 2 — 5-stage worker loop (the core)
- `agent/worker.py` (Think→Plan→Act→Observe→Verify), `protocol.py` (`tool_call`/`FINAL_ANSWER`
  parser), `verifier.py`, `self_repair.py`, `context_compressor.py` (compress at 92%),
  `tools/` (Tool ABC, `bash`, file ops), and `infrastructure/execution_sandbox.py`.
- **Verify (Done-when):** `task "what is 17! mod 1000?"` drives **one bash call + a deterministic
  Verify** and returns the correct computed value. *(See §4 — the spec's literal `880` is
  arithmetically wrong; the agent computes and verifies the true value `0`.)* Plus tool-dispatch,
  verify-fail→self-repair→retry, and max-turns guard tests. 7 worker tests pass.

### Phase 3 — Hierarchical task graph
- `task_graph/` (node/graph/scheduler/verifier) + `agent/task_graph_builder.py` +
  `agent/command_center.py`. Ready nodes dispatch in parallel (capped by `swarm.max_workers`),
  ordered by `PriorityScheduler`; a failed node cascades only to its dependents.
- **Verify (Done-when):** independent verification — a 4-node graph with one failing node leaves
  independent branches `DONE` while only the dependent chain `FAILED`. 6 tests pass.

### Phase 6 — Guardian + file-level audit
- `governance/` (guardian_agent, permission, audit_logger, + moved safety/injection-guard copies).
  Every `bash`/`write_file` is risk-classified; `rm -rf`/`mkfs`/fork-bombs/etc. are `HIGH` and
  refused without an explicit allow or interactive approval; all actions are appended to a
  JSON-lines audit log supporting `replay`/`rollback`.
- **Verify (Done-when):** independent verification — `rm -rf /` blocked (`HIGH`), `echo` allowed,
  `file_write` audited; audit replay recovers the action sequence. 16 tests pass.

### Phase 4 — Dynamic tool forge
- `agent/tools/tool_forge.py` + `tool_library.py`. Vector/keyword search the library; if absent,
  the LLM writes a script tool, it is tested in the sandbox, then registered for reuse.
- **Verify (Done-when):** independent verification — forge writes+tests+registers a tool, a similar
  request **reuses** it (no second generation call), library size stays 1. 3 tests pass.

### Phase 5 / 7 — interfaces & polish
- Web tools (`web_search`/`web_fetch`, keyless DuckDuckGo + prompt-injection-guarded), MCP client
  (`fastmcp` seam, degrades cleanly when absent), computer-use (Playwright bridge),
  `infrastructure/cloud_sandbox.py` (E2B seam). CLI: `--graph` (DAG), `resume` (audit replay),
  cost + diagnosis reporting; `agent/priority_scheduler.py` + `agent/diagnoser.py`.
- **Key-gated (clearly marked):** E2B cloud VMs and live MCP servers need `E2B_API_KEY` /
  `fastmcp` + a server; without them the sandbox falls back to docker→local and MCP returns a
  "not configured" result. Docker **is** available here, so execution is real.

---

## 4. Spec discrepancies found and how they were resolved (§11.1)

1. **`17! mod 1000` = 0, not 880.** §7 asserts the answer is `880`, but `17!` has three trailing
   zeros (⌊17/5⌋ = 3), so `17! mod 1000 = 0`. Hardcoding `880` would fabricate a result, violating
   the agent's own contract ("never claim a result you did not produce"). The loop computes and
   verifies the **true** value; the test asserts the independently-computed answer.
2. **§5 bugs 1–3 already fixed / bug 4 file absent** (see §2.1). Not re-edited; locked with tests.
3. **`HypothesisPool → TaskGraph` rename collides with a *new* `task_graph/` DAG.** §3 also says
   "keep `core/hypothesis.py`." Resolution: keep `HypothesisPool` for the preserved research tool;
   the **new** DAG is `TaskGraph` in `task_graph/`. No breakage.
4. **§1.4 rename map vs. §0/§3 "preserve the pipeline."** A literal in-place rename
   (`ReflectionAgent→VerifierAgent`, `SupervisorAgent→CommandCenter`, …) would break the preserved
   orchestrator. Resolution: the **new names exist as the new JARVIS-layer components**
   (`VerifierAgent`, `SelfRepairAgent`, `CommandCenter`, `DynamicToolForge`, `GuardianAgent`,
   `ExecutionSandbox`, plus `PriorityScheduler`/`DiagnoseAgent` from the §3 layout); the old
   research agents are kept. `ProximityAgent` (map says "remove") is **kept** because the preserved
   pipeline imports it — removing it would contradict §0/§3.
5. **`.env` key wiring.** §6 documents `OPENROUTER_API_KEY`, but the router only read `LLM_API_KEY`.
   Added a minimal env fallback in the OpenRouter branch (mirroring the existing Anthropic branch).

---

## 5. Test results

- **Before:** 0 tests runnable (collection error).
- **After:** **79 passed, 1 skipped** (the skipped module is the legacy climate test, whose
  `climate_models` module was removed before this work). `ruff check` clean on the new code.
- All Phase "Done-when" gates and all confirmed-finding fixes were additionally **verified
  independently** (with different scenarios than the agents' own tests) before being accepted.

```
tests/agent/test_worker.py ............  8   (incl. 17!-style one-bash-call + verify, node-verify)
tests/test_governance.py .............. 30   (rm-rf + classifier bypasses blocked, audit replay)
tests/agent/test_task_graph.py ........  6   (parallel DAG + fault isolation)
tests/agent/test_sandbox_and_intent.py.  8   (incl. env-scrub no-secret-leak)
tests/test_known_bugs.py ..............  8   (regressions for §5 bugs 1,2,3,5)
tests/agent/test_tool_forge.py ........  4   (forge → test → register → reuse, entrypoint injection)
tests/agent/test_web_tools.py .........  3   (injection-guarded web search/fetch)
tests/agent/test_ssrf_and_injection.py.  5   (metadata/loopback blocked, envelope neutralised)
tests/agent/test_priority_and_diagnose.  3
```

---

## 6. How to run (verified)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q              # 79 passed, 1 skipped
cp .env.example .env                        # add OPENROUTER_API_KEY (free Kimi K2.6) — or use Ollama
.venv/bin/co-scientist task "compute 2^64 and write it to power.txt"
.venv/bin/co-scientist research "novel drug targets for Alzheimer's disease"   # preserved pipeline
```

---

## 6a. Adversarial review & hardening pass

After the build, a multi-angle review (4 dimensions × parallel finders, each finding
adversarially re-verified by an independent agent) was run over the new JARVIS layer.
It confirmed **27 real findings**; the high-severity ones were fixed:

**Guardian was bypassed on several execution paths (§4/§10 violations) — fixed:**
- `VerifierAgent` ran the model-supplied `verification` command ungated → now Guardian-gated + audited.
- `CommandCenter` (DAG path) built Workers with no guardian/audit → now every node Worker is gated + audited.
- `DynamicToolForge` ran LLM-authored scripts directly → now gated through `ctx.guardian` + audited.
- `EditTool` mutated files without a check (unlike `WriteTool`) → now gated.

**Correctness — fixed:**
- **Parallel budget collapse (#0):** every Worker read the *global* shared cost tracker against a
  *per-worker* budget, so the first node to cross the cap stopped all of them. Now each Worker
  measures spend as a **delta from its own start**.
- **Dropped node verification (#1):** the planner's deterministic `node.verification` was never run.
  `CommandCenter` now threads it into the Worker's Verify stage (authoritative over the model's claim);
  regression test asserts a node cannot "succeed" past a failing check.

**Security hardening — fixed:**
- Command classifier now escalates pipes/redirects/subshells/chaining/nested-interpreters and
  `~`/`$HOME`/`../` traversal (closes `cat ~/.ssh/...`, `echo x > /etc/passwd` bypasses).
- Local sandbox no longer inherits the host environment (was leaking `*_API_KEY`/`*_TOKEN`); env is scrubbed.
- SSRF guard on `web_fetch`/`computer_use` (rejects loopback/private/link-local/`169.254.169.254`).
- Tool-forge entrypoint is now a fixed per-language template (no LLM-authored shell), and the
  injection guard escapes the source URL and strips envelope-closing tags from untrusted content.
- Audit log redacts secret-looking *values* (not just key names); Docker timeout now kills the container.

**Simplicity cleanups:** removed dead `pick_ready`, duplicate `_LLMRepair`, unused `to_dict/from_dict`;
fixed `rollback()` negative-index handling.

Each fix is covered by a regression test; the full suite remains green after hardening.

## 7. What a user should know about boundaries

- **Needs an LLM key to *act*.** The loop logic, sandbox, DAG, forge, and guardian are fully real
  and tested deterministically, but driving a live task needs an LLM (free OpenRouter Kimi, or a
  local Ollama for zero-key use).
- **E2B / live MCP are key-gated seams**, implemented as interfaces with graceful fallback; Docker
  sandboxing is real and used by default when present.
- The research pipeline's heavy ML extras (`chromadb`, `sentence-transformers`, `hdbscan`) remain
  lazy-loaded; the JARVIS core runs without them.
