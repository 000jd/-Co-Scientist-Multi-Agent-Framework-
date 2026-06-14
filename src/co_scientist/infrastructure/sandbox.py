"""Docker Sandbox for secure execution of code."""

import tempfile
import os
from typing import Dict, Optional
from pydantic import BaseModel
from co_scientist.core.config import AgentConfig

class SandboxResult(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float
    exit_code: int = 0

class DockerSandbox:
    """Docker-based sandboxed code execution.
    
    Fix #2: Default memory limit is 2GB for data analysis.
    """
    
    def __init__(self, config: AgentConfig):
        self.memory_limit = config.docker_memory_limit  # "2g"
        self.cpu_limit = config.docker_cpu_limit
        self.timeout = config.docker_timeout_seconds
        self.image = config.docker_image
        
    async def execute(
        self,
        code: str,
        memory_limit: Optional[str] = None,
        timeout: Optional[int] = None,
        input_files: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """Execute Python code in Docker sandbox."""
        import time
        start = time.time()
        mem = memory_limit or self.memory_limit
        tout = timeout or self.timeout
        
        try:
            import docker
            client = docker.from_env()
        except Exception as e:
            # Fallback to local execution if Docker unavailable
            return SandboxResult(
                success=False,
                output="",
                error=f"Docker unavailable: {e}",
                execution_time=time.time() - start,
            )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            
            # Write input files
            if input_files:
                for name, content in input_files.items():
                    with open(os.path.join(tmpdir, name), "w") as f:
                        f.write(content)
            
            container = None
            try:
                # Ensure image exists
                try:
                    client.images.get(self.image)
                except docker.errors.ImageNotFound:
                    client.images.pull(self.image)

                container = client.containers.run(
                    self.image,
                    command=["python", "/workspace/script.py"],
                    volumes={tmpdir: {"bind": "/workspace", "mode": "ro"}},
                    working_dir="/workspace",
                    detach=True,
                    mem_limit=mem,  # Fix #2: 2GB
                    cpu_count=self.cpu_limit,
                    network_disabled=True,
                )
                
                result = container.wait(timeout=tout)
                
                # Truncate logs
                stdout_bytes = container.logs(stdout=True, stderr=False)
                stderr_bytes = container.logs(stdout=False, stderr=True)
                stdout = stdout_bytes.decode("utf-8")[-10000:] if stdout_bytes else ""
                stderr = stderr_bytes.decode("utf-8")[-10000:] if stderr_bytes else ""
                
                exit_code = result.get("StatusCode", 1)
                
                # OOM Detection
                if exit_code == 137:
                    stderr = f"[OOM Error] Process killed. Memory limit ({mem}) exceeded.\n" + stderr
                
                return SandboxResult(
                    success=exit_code == 0,
                    output=stdout,
                    error=stderr if exit_code != 0 else None,
                    execution_time=time.time() - start,
                    exit_code=exit_code,
                )
            except Exception as e:
                return SandboxResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=time.time() - start,
                )
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
