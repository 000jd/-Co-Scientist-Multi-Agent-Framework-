"""ExecutionSandbox (local provider) + IntentParser routing + §6 config keys."""

import pytest

from co_scientist.core.config import SandboxConfig, load_config
from co_scientist.infrastructure.execution_sandbox import ExecutionSandbox
from co_scientist.agent.intent_parser import IntentParser


@pytest.mark.asyncio
async def test_local_sandbox_runs_shell(tmp_path):
    sb = ExecutionSandbox(SandboxConfig(provider="local"), workspace_root=str(tmp_path))
    assert sb.provider == "local"
    result = await sb.run("echo hello && expr 6 \\* 7", workdir=str(tmp_path))
    assert result.success
    assert "hello" in result.stdout
    assert "42" in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_nonzero_exit(tmp_path):
    sb = ExecutionSandbox(SandboxConfig(provider="local"), workspace_root=str(tmp_path))
    result = await sb.run("exit 3", workdir=str(tmp_path))
    assert not result.success
    assert result.exit_code == 3


@pytest.mark.asyncio
async def test_local_sandbox_timeout(tmp_path):
    sb = ExecutionSandbox(
        SandboxConfig(provider="local", local_timeout_seconds=1),
        workspace_root=str(tmp_path),
    )
    result = await sb.run("sleep 5", workdir=str(tmp_path))
    assert not result.success
    assert result.exit_code == 124


@pytest.mark.asyncio
async def test_intent_routes_research():
    parser = IntentParser(llm_router=None)
    intent = await parser.parse("Generate novel drug targets for Alzheimer's disease")
    assert intent.kind == "research"


@pytest.mark.asyncio
async def test_intent_defaults_to_task():
    parser = IntentParser(llm_router=None)
    intent = await parser.parse("render floor-plan.pdf as a walkthrough -> ./out.mp4")
    assert intent.kind == "task"
    assert intent.constraints.get("output_path") == "./out.mp4"


@pytest.mark.asyncio
async def test_local_sandbox_does_not_leak_host_secrets(tmp_path, monkeypatch):
    secret = "sk-leak-canary-1234567890"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    sb = ExecutionSandbox(SandboxConfig(provider="local"), workspace_root=str(tmp_path))
    result = await sb.run("env", workdir=str(tmp_path))
    assert result.success
    assert "OPENROUTER_API_KEY" not in result.stdout
    assert secret not in result.stdout


@pytest.mark.asyncio
async def test_local_sandbox_passes_through_nonsecret_env(tmp_path):
    sb = ExecutionSandbox(SandboxConfig(provider="local"), workspace_root=str(tmp_path))
    result = await sb.run("echo $MY_FLAG", workdir=str(tmp_path), env={"MY_FLAG": "on"})
    assert result.success
    assert "on" in result.stdout


def test_config_has_jarvis_sections():
    cfg = load_config("config.yaml")
    assert cfg.agent.max_turns == 50
    assert cfg.agent.budget_usd == 5.0
    assert cfg.sandbox.provider in ("e2b", "docker", "local")
    assert cfg.tool_forge.max_repair_attempts == 5
    assert cfg.swarm.max_workers == 10
