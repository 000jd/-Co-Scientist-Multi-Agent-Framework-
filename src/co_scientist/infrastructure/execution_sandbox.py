"""Unified execution sandbox — the single substrate for the Act stage (CLAUDE.md §1.1).

Replaces/expands the Python-only ``DockerSandbox`` with an arbitrary-shell-command
runner across three providers (CLAUDE.md §1.4, §2):

    e2b    — cloud micro-VM (requires E2B_API_KEY + the ``e2b`` SDK)
    docker — local container (``coscientist-sandbox`` image), the safe default
    local  — host subprocess in a project-scoped workdir (no-Docker fallback)

The configured provider is resolved with graceful fallback (e2b → docker → local)
so ``co-scientist task`` works out of the box on a laptop while still preferring
real isolation when it is available.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Optional

import structlog
from pydantic import BaseModel

log = structlog.get_logger("infrastructure.execution_sandbox")

_MAX_CAPTURE = 20_000  # truncate captured stdout/stderr to this many trailing chars


class ExecResult(BaseModel):
    """Outcome of a single sandboxed command."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    provider: str = "local"
    error: Optional[str] = None

    @property
    def output(self) -> str:
        """Combined human-readable output (stdout first, then stderr)."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr if self.success else f"[stderr]\n{self.stderr}")
        return "\n".join(parts).strip()


def _truncate(text: str) -> str:
    return text[-_MAX_CAPTURE:] if len(text) > _MAX_CAPTURE else text


def _is_sensitive(name: str) -> bool:
    """True if an env var name looks like a secret that must not leak to the agent."""
    upper = name.upper()
    if upper.endswith(("_API_KEY", "_TOKEN", "_SECRET", "KEY")):
        return True
    return upper.startswith(("AWS_", "OPENAI_", "ANTHROPIC_", "OPENROUTER_"))


def _scrubbed_env(home: str, extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Build a minimal, secret-free environment for host subprocesses.

    Starts from a minimal base (PATH from the host so python3/bash are found, plus
    LANG/LC_ALL/TZ if present), sets HOME to the workspace dir, then layers the
    caller-supplied ``extra`` dict on top. Host secrets (``*_API_KEY``, ``*_TOKEN``,
    ``*_SECRET``, ``*KEY``, ``AWS_*``, ``OPENAI_*``, ``ANTHROPIC_*``,
    ``OPENROUTER_*``) are never inherited from ``os.environ``.
    """
    base: Dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": home,
    }
    for key in ("LANG", "LC_ALL", "TZ"):
        val = os.environ.get(key)
        if val is not None:
            base[key] = val
    for key, val in (extra or {}).items():
        if _is_sensitive(key):
            continue
        base[key] = val
    return base


def _docker_available() -> bool:
    try:
        import docker  # noqa: F401

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _e2b_available(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    try:
        import e2b_code_interpreter  # noqa: F401

        return True
    except Exception:
        try:
            import e2b  # noqa: F401

            return True
        except Exception:
            return False


class ExecutionSandbox:
    """Runs shell commands in the configured (or best-available) sandbox provider."""

    def __init__(self, sandbox_config, workspace_root: str = "./projects"):
        self.config = sandbox_config
        self.workspace_root = Path(workspace_root)
        self._e2b_key = None
        if getattr(sandbox_config, "e2b_api_key", None) is not None:
            self._e2b_key = sandbox_config.e2b_api_key.get_secret_value()
        self._e2b_key = self._e2b_key or os.environ.get("E2B_API_KEY")
        self.provider = self._resolve_provider(
            getattr(sandbox_config, "provider", "docker")
        )
        log.info(
            "execution_sandbox.ready",
            requested=getattr(sandbox_config, "provider", "docker"),
            resolved=self.provider,
        )

    # ── provider resolution ────────────────────────────────────────────────
    def _resolve_provider(self, requested: str) -> str:
        """Pick the first usable provider, preferring the requested one."""
        order = [requested]
        for fallback in ("docker", "local"):
            if fallback not in order:
                order.append(fallback)
        for p in order:
            if p == "e2b" and _e2b_available(self._e2b_key):
                return "e2b"
            if p == "docker" and _docker_available():
                return "docker"
            if p == "local":
                return "local"
            if p not in ("e2b", "docker", "local"):
                log.warning("execution_sandbox.unknown_provider", provider=p)
        return "local"

    # ── public API ─────────────────────────────────────────────────────────
    async def run(
        self,
        command: str,
        workdir: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """Execute ``command`` as a shell command and return its captured result."""
        work = Path(workdir) if workdir else self.workspace_root
        work.mkdir(parents=True, exist_ok=True)
        if self.provider == "e2b":
            return await self._run_e2b(command, work, timeout, env)
        if self.provider == "docker":
            return await self._run_docker(command, work, timeout, env)
        return await self._run_local(command, work, timeout, env)

    # ── local subprocess (no-Docker fallback; project-scoped) ──────────────
    async def _run_local(self, command, work: Path, timeout, env) -> ExecResult:
        start = time.time()
        tout = timeout or getattr(self.config, "local_timeout_seconds", 120)
        full_env = _scrubbed_env(str(work.resolve()), env)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(work),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=tout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecResult(
                    success=False,
                    exit_code=124,
                    provider="local",
                    execution_time=time.time() - start,
                    error=f"timeout after {tout}s",
                )
            code = proc.returncode or 0
            return ExecResult(
                success=code == 0,
                stdout=_truncate(out.decode("utf-8", "replace")),
                stderr=_truncate(err.decode("utf-8", "replace")),
                exit_code=code,
                execution_time=time.time() - start,
                provider="local",
                error=None if code == 0 else f"exit code {code}",
            )
        except Exception as e:  # noqa: BLE001
            return ExecResult(
                success=False,
                exit_code=1,
                provider="local",
                execution_time=time.time() - start,
                error=str(e),
            )

    # ── docker container ───────────────────────────────────────────────────
    async def _run_docker(self, command, work: Path, timeout, env) -> ExecResult:
        return await asyncio.to_thread(
            self._run_docker_sync, command, work, timeout, env
        )

    def _run_docker_sync(self, command, work: Path, timeout, env) -> ExecResult:
        start = time.time()
        tout = timeout or getattr(self.config, "e2b_timeout_seconds", 300)
        image = getattr(self.config, "docker_image", "coscientist-sandbox:latest")
        try:
            import docker
        except Exception as e:  # noqa: BLE001
            return ExecResult(
                success=False,
                exit_code=1,
                provider="docker",
                execution_time=time.time() - start,
                error=f"docker SDK unavailable: {e}",
            )
        # Reject symlink/parent escapes: the rw bind mount must stay inside
        # the configured workspace_root (finding #16).
        resolved_work = work.resolve()
        root = self.workspace_root.resolve()
        if resolved_work != root and root not in resolved_work.parents:
            return ExecResult(
                success=False,
                exit_code=1,
                provider="docker",
                execution_time=time.time() - start,
                error=f"workdir {resolved_work} escapes workspace_root {root}",
            )
        container = None
        try:
            client = docker.from_env()
            try:
                client.images.get(image)
            except Exception:
                image = (
                    "python:3.11-slim"  # ship-anywhere default if custom image absent
                )
                try:
                    client.images.get(image)
                except Exception:
                    client.images.pull(image)
            container = client.containers.run(
                image,
                command=["bash", "-lc", command],
                volumes={str(resolved_work): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                environment=env or {},
                user="1000:1000",  # non-root inside the container (finding #16)
                detach=True,
                mem_limit=getattr(self.config, "docker_memory_limit", "2g"),
                nano_cpus=int(getattr(self.config, "docker_cpu_limit", 2.0) * 1e9),
            )
            try:
                result = container.wait(timeout=tout)
            except Exception:  # noqa: BLE001
                # container.wait timed out (finding #22): kill before remove so we
                # never leave a runaway container behind.
                try:
                    container.kill()
                except Exception:
                    pass
                return ExecResult(
                    success=False,
                    exit_code=124,
                    provider="docker",
                    execution_time=time.time() - start,
                    error=f"timeout after {tout}s",
                )
            code = result.get("StatusCode", 1)
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", "replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", "replace"
            )
            if code == 137:
                stderr = f"[OOM] memory limit exceeded\n{stderr}"
            return ExecResult(
                success=code == 0,
                stdout=_truncate(stdout),
                stderr=_truncate(stderr),
                exit_code=code,
                execution_time=time.time() - start,
                provider="docker",
                error=None if code == 0 else f"exit code {code}",
            )
        except Exception as e:  # noqa: BLE001
            return ExecResult(
                success=False,
                exit_code=1,
                provider="docker",
                execution_time=time.time() - start,
                error=str(e),
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    # ── e2b cloud VM (key-gated; falls back inside resolve) ────────────────
    async def _run_e2b(self, command, work: Path, timeout, env) -> ExecResult:
        start = time.time()
        tout = timeout or getattr(self.config, "e2b_timeout_seconds", 300)
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore

            def _exec():
                sbx = Sandbox(api_key=self._e2b_key)
                try:
                    res = sbx.commands.run(command, timeout=tout, envs=env or {})
                    return ExecResult(
                        success=res.exit_code == 0,
                        stdout=_truncate(res.stdout or ""),
                        stderr=_truncate(res.stderr or ""),
                        exit_code=res.exit_code,
                        execution_time=time.time() - start,
                        provider="e2b",
                        error=None
                        if res.exit_code == 0
                        else f"exit code {res.exit_code}",
                    )
                finally:
                    sbx.kill()

            return await asyncio.to_thread(_exec)
        except Exception as e:  # noqa: BLE001
            log.warning("execution_sandbox.e2b_failed_fallback_local", error=str(e))
            return await self._run_local(command, work, timeout, env)


def make_sandbox(config, workspace_root: str = "./projects") -> ExecutionSandbox:
    """Construct an ExecutionSandbox from a full Config or a bare SandboxConfig."""
    sandbox_cfg = getattr(config, "sandbox", config)
    return ExecutionSandbox(sandbox_cfg, workspace_root=workspace_root)
