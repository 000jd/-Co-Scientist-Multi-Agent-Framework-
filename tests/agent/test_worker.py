"""Phase 2 acceptance + unit tests for the 5-stage worker loop (CLAUDE.md §7).

These drive the REAL Worker + REAL ExecutionSandbox (local provider) with a scripted
mock LLM — no API key, fully deterministic.
"""

import json
from pathlib import Path

import pytest

from co_scientist.agent.worker import Worker
from co_scientist.agent.protocol import parse_action
from co_scientist.core.config import SandboxConfig
from co_scientist.infrastructure.execution_sandbox import ExecutionSandbox
from tests.mock_llm import MockLLMRouter


def _local_sandbox(tmp_path: Path) -> ExecutionSandbox:
    cfg = SandboxConfig(provider="local", local_timeout_seconds=30)
    return ExecutionSandbox(cfg, workspace_root=str(tmp_path))


# ── protocol ───────────────────────────────────────────────────────────────
def test_parse_tool_call():
    a = parse_action(
        'thinking... tool_call({"tool": "bash", "args": {"command": "echo hi"}})'
    )
    assert a.kind == "tool_call"
    assert a.tool == "bash"
    assert a.args["command"] == "echo hi"


def test_parse_final_answer_wins_over_tool_call():
    a = parse_action(
        'tool_call({"tool":"bash","args":{}}) FINAL_ANSWER({"answer":"42"})'
    )
    assert a.kind == "final_answer"
    assert a.answer == "42"


def test_parse_nested_parens_in_command():
    a = parse_action(
        'tool_call({"tool": "bash", "args": {"command": "python3 -c \\"print((1+2))\\""}})'
    )
    assert a.kind == "tool_call"
    assert "print((1+2))" in a.args["command"]


# ── Phase 2 "Done when": `task "17! mod 1000?"` → bash call + deterministic verify ──
# NOTE: CLAUDE.md §7 states the expected answer is 880, but that is arithmetically
# wrong: 17! = 355_687_428_096_000 has 3 trailing zeros (floor(17/5)=3), so
# 17! mod 1000 == 0. This test validates the spec's *headline behaviour* (one bash
# call + a deterministic Verify produces the correct computed value) using the true
# answer, computed independently rather than hardcoded. (Recorded in the audit.)
@pytest.mark.asyncio
async def test_factorial_mod_one_bash_call_plus_verify(tmp_path):
    import math

    true_answer = str(math.factorial(17) % 1000)  # "0"
    compute = "python3 -c 'import math; print(math.factorial(17) % 1000)'"
    plan1 = (
        "I will compute it.\ntool_call("
        + json.dumps({"tool": "bash", "args": {"command": compute}})
        + ")"
    )
    final = (
        "FINAL_ANSWER("
        + json.dumps(
            {
                "answer": true_answer,
                "verification": f'test "$({compute})" = {true_answer}',
            }
        )
        + ")"
    )
    llm = MockLLMRouter(responses=[plan1, final])
    worker = Worker(
        llm_router=llm,
        sandbox=_local_sandbox(tmp_path),
        max_turns=10,
        project_id="t1",
        workdir=str(tmp_path / "t1"),
    )

    result = await worker.run("what is 17! mod 1000?")

    assert result.success is True
    assert result.answer.strip() == true_answer
    assert result.verified is True
    assert (
        result.verification_method == "command"
    )  # deterministic shell verify, no LLM judge
    assert result.stop_reason == "final_answer"
    assert result.turns == 2


# ── tool dispatch + observation flow ────────────────────────────────────────
@pytest.mark.asyncio
async def test_bash_tool_executes_and_observes(tmp_path):
    llm = MockLLMRouter(
        responses=[
            'tool_call({"tool": "bash", "args": {"command": "echo HELLO_WORLD"}})',
            'FINAL_ANSWER({"answer": "done"})',
        ]
    )
    worker = Worker(
        llm_router=llm,
        sandbox=_local_sandbox(tmp_path),
        max_turns=5,
        project_id="t2",
        workdir=str(tmp_path / "t2"),
    )
    result = await worker.run("print hello")
    # the observation from the bash call must have entered the transcript
    transcript = "\n".join(m["content"] for m in result.transcript)
    assert "HELLO_WORLD" in transcript


# ── verify-fail → self-repair → retry ───────────────────────────────────────
@pytest.mark.asyncio
async def test_verify_failure_triggers_self_repair_then_succeeds(tmp_path):
    bad_check = 'FINAL_ANSWER({"answer": "wrong", "verification": "false"})'
    good = 'FINAL_ANSWER({"answer": "right", "verification": "true"})'
    plans = iter([bad_check, good])

    def route(messages, rf):
        # Structured sub-agent calls (self-repair/verify-judge) get a benign object so
        # they don't consume the scripted plan queue.
        if rf is not None:
            return {
                n: (True if f.annotation is bool else "ok")
                for n, f in rf.model_fields.items()
            }
        return next(plans)

    llm = MockLLMRouter(router=route)
    worker = Worker(
        llm_router=llm,
        sandbox=_local_sandbox(tmp_path),
        max_turns=5,
        project_id="t3",
        workdir=str(tmp_path / "t3"),
    )
    result = await worker.run("produce the right answer")
    assert result.success is True
    assert result.answer == "right"
    assert result.turns == 2  # first FINAL failed verify, second passed
    # a self-repair diagnosis must have been injected
    assert any("[VERIFY FAILED]" in m["content"] for m in result.transcript)


# ── budget / max_turns guards ───────────────────────────────────────────────
@pytest.mark.asyncio
async def test_node_verification_is_enforced_over_model_claim(tmp_path):
    """A planner-supplied node verification is authoritative: the worker cannot
    'succeed' by emitting FINAL_ANSWER if the node's deterministic check fails (review #1)."""
    # model claims done with NO verification of its own
    llm = MockLLMRouter(router=lambda m, rf: 'FINAL_ANSWER({"answer": "claimed done"})')
    worker = Worker(
        llm_router=llm,
        sandbox=_local_sandbox(tmp_path),
        max_turns=3,
        project_id="nv",
        workdir=str(tmp_path / "nv"),
    )
    failed = await worker.run("do it", verification="false")  # node check always fails
    assert failed.success is False  # model's claim is overridden by the node check

    llm2 = MockLLMRouter(responses=['FINAL_ANSWER({"answer": "really done"})'])
    worker2 = Worker(
        llm_router=llm2,
        sandbox=_local_sandbox(tmp_path),
        max_turns=3,
        project_id="nv2",
        workdir=str(tmp_path / "nv2"),
    )
    ok = await worker2.run("do it", verification="true")  # node check passes
    assert ok.success is True and ok.verified is True


@pytest.mark.asyncio
async def test_max_turns_terminates(tmp_path):
    # never emits FINAL_ANSWER → must stop at max_turns
    llm = MockLLMRouter(
        router=lambda msgs, rf: (
            'tool_call({"tool": "bash", "args": {"command": "true"}})'
        )
    )
    worker = Worker(
        llm_router=llm,
        sandbox=_local_sandbox(tmp_path),
        max_turns=3,
        project_id="t4",
        workdir=str(tmp_path / "t4"),
    )
    result = await worker.run("loop forever")
    assert result.success is False
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
