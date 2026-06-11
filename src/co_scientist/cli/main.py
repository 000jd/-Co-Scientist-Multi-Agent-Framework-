"""
Co-Scientist CLI - Command-line interface for the multi-agent system.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from co_scientist.core.config import Config, load_config
from co_scientist.core.orchestrator import CoScientistOrchestrator

app = typer.Typer(
    name="co-scientist",
    help="Co-Scientist Multi-Agent Framework for Autonomous Scientific Discovery",
    rich_markup_mode="rich",
)
console = Console()


def get_config_path() -> Optional[str]:
    """Find configuration file."""
    for path in ["config.yaml", "config.yml", "config/config.yaml"]:
        if Path(path).exists():
            return path
    return None


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Co-Scientist - Autonomous Scientific Discovery with AI Agents."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose


@app.command()
def research(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Research goal"),
    domain: str = typer.Option("generic", "--domain", "-d", help="Research domain"),
    iterations: int = typer.Option(1, "--iterations", "-i", help="Number of iterations"),
    closed_loop: bool = typer.Option(False, "--closed-loop", "-l", help="Enable closed-loop (validation)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
):
    """Run a research cycle with Co-Scientist agents."""
    config_path = ctx.obj.get("config_path") or get_config_path()
    verbose = ctx.obj.get("verbose", False)

    config = load_config(config_path)
    if verbose:
        config.debug = True
        
    orchestrator = CoScientistOrchestrator(config, config.get_candidate_model_class())
    
    console.print(f"\n[bold green]Starting Co-Scientist Pipeline...[/]")
    console.print(f"[bold cyan]Goal:[/] {goal}")
    console.print(f"[bold cyan]Iterations:[/] {iterations}\n")
    
    async def _run():
        from co_scientist.core.events import EventType
        async def log_event(event):
            time_str = event.timestamp.strftime('%H:%M:%S')
            if event.event_type == EventType.HYPOTHESIS_GENERATED:
                console.print(f"[[dim]{time_str}[/]] [bold green]Hypothesis Generated[/] [dim]({event.source})[/]")
            elif event.event_type == EventType.AGENT_STARTED:
                console.print(f"[[dim]{time_str}[/]] [bold blue]Agent Started:[/] {event.source}")
            elif event.event_type == EventType.AGENT_COMPLETED:
                console.print(f"[[dim]{time_str}[/]] [bold blue]Agent Completed:[/] {event.source}")
            elif event.event_type == EventType.HYPOTHESIS_RANKED:
                console.print(f"[[dim]{time_str}[/]] [bold yellow]Tournament Debate Concluded[/]")
            elif event.event_type == EventType.SAFETY_FLAG_RAISED:
                console.print(f"[[dim]{time_str}[/]] [bold red on white]SAFETY FLAG RAISED![/]")
            else:
                console.print(f"[[dim]{time_str}[/]] [cyan]{event.event_type.value}[/] [dim][{event.source}][/]")
                
        for e in EventType:
            await orchestrator.event_bus.subscribe(e, log_event)
            
        await orchestrator.run_pipeline(goal, iterations)
        
    try:
        asyncio.run(_run())
        console.print("\n[bold cyan]Research Complete![/bold cyan]")
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Error: {e}[/bold red]")
        import traceback
        if verbose:
            console.print(traceback.format_exc())
            
    lb = orchestrator.leaderboard.to_display()
    output_path = Path(output) if output else Path("data/results/latest_run.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lb, indent=2, default=str))
    console.print(f"\n[bold green]Results saved to:[/] {output_path.absolute()}")
        

@app.command()
def discover(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Research goal"),
    until_convergence: bool = typer.Option(True, "--converge", "-c", help="Run until convergence"),
    max_cycles: Optional[int] = typer.Option(None, "--max-cycles", help="Safety cap (None = infinite)"),
):
    """Unlimited discovery mode with convergence detection."""
    from co_scientist.core.task_manager import TaskManager
    
    config = load_config(ctx.obj.get("config_path"))
    manager = TaskManager(config, max_cycles=max_cycles)
    
    try:
        result = asyncio.run(manager.run(goal))
        console.print(f"\n[bold green]Discovery Complete![/]")
        console.print(f"Cycles: {result['cycles_run']}")
        console.print(f"Hypotheses generated: {result['total_hypotheses']}")
        console.print(f"Converged: {result['converged']}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Check checkpoints.[/]")
        manager.stop()


@app.command()
def leaderboard(
    top: int = typer.Option(20, "--top", "-n", help="Show top N"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Show current hypothesis leaderboard."""
    async def _leaderboard():
        from co_scientist.core.database import DatabaseManager, HypothesisModel
        cfg = load_config(config)
        db = DatabaseManager(cfg.database.url)
        await db.init_db()

        async with db.async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(HypothesisModel).order_by(HypothesisModel.created_at.desc())
            )
            models = result.scalars().all()

        if not models:
            console.print("[yellow]No hypotheses ranked yet. Run research first.[/yellow]")
            return

        from co_scientist.ranking.trueskill_adapter import CombinedLeaderboard
        lb_obj = CombinedLeaderboard()
        for m in models:
            data = m.data if isinstance(m.data, dict) else json.loads(m.data)
            lb_obj.add_or_update(
                hypothesis_id=m.id,
                title=m.title,
                trueskill_mu=data.get("trueskill_rating", {}).get("mu", 25.0),
                trueskill_sigma=data.get("trueskill_rating", {}).get("sigma", 8.333),
                debate_wins=data.get("debate_wins", 0),
                debate_losses=data.get("debate_losses", 0),
                epistemic_uncertainty_score=0.5
            )

        lb = lb_obj.to_display()

        if not lb:
            console.print("[yellow]No hypotheses ranked yet. Run research first.[/yellow]")
            return

        table = Table(title="Hypothesis Leaderboard", box=box.DOUBLE_EDGE)
        table.add_column("Rank", justify="right", style="bold")
        table.add_column("ID", style="dim")
        table.add_column("Title")
        table.add_column("ELO", justify="right", style="yellow")
        table.add_column("W/L/D", justify="center")
        table.add_column("Tier", style="cyan")

        for entry in lb:
            wld = f"{entry.get('wins', 0)}/{entry.get('losses', 0)}/{entry.get('draws', 0)}"
            table.add_row(
                str(entry.get("rank", "")),
                entry.get("id", "")[:8],
                entry.get("title", "")[:55],
                str(entry.get("elo", "")),
                wld,
                entry.get("tier", ""),
            )

        console.print(table)

    asyncio.run(_leaderboard())


@app.command()
def agents():
    """List all available agents and their descriptions."""
    table = Table(title="Co-Scientist Agents", box=box.ROUNDED)
    table.add_column("Agent", style="bold cyan")
    table.add_column("Type", style="blue")
    table.add_column("Description", style="green")

    agents_info = [
        ("Supervisor", "orchestrator", "Top-level orchestrator. Parses goals, configures pipeline, manages workers."),
        ("Generation", "co-scientist", "Brainstorms ideas, searches literature, generates hypotheses."),
        ("Reflection", "co-scientist", "Critical reviewer. Fact-checks, evaluates novelty, finds flaws."),
        ("Proximity", "co-scientist", "Maps hypotheses to vector space, detects redundancy."),
        ("Evolution", "co-scientist", "Refines hypotheses, combines ideas, bridges logical gaps."),
        ("Ranking", "co-scientist", "ELO tournament system. Head-to-head debates with AI judge."),
        ("Meta-review", "co-scientist", "Cross-agent evaluation, final quality assessment."),
        ("Validation", "validation", "Runs literature, simulation, and consensus checks."),
    ]

    for name, type_, desc in agents_info:
        table.add_row(name, type_, desc)

    console.print(table)
    console.print("\n[dim]Co-Scientist agents form the hypothesis generation pipeline.[/dim]")


@app.command()
def config_show(
    output: Optional[str] = typer.Option(None, "--output", "-o"),
):
    """Show current configuration."""
    cfg = load_config()

    config_dict = {}
    for section in ["llm", "search", "domain", "agents", "database", "experiment"]:
        obj = getattr(cfg, section, None)
        if obj:
            config_dict[section] = obj.model_dump() if hasattr(obj, "model_dump") else str(obj)

    config_json = json.dumps(config_dict, indent=2, default=str)
    console.print(Panel(config_json, title="Configuration", border_style="blue"))

    if output:
        Path(output).write_text(config_json)
        console.print(f"\n[green]Config saved to: {output}[/green]")


@app.command()
def init(
    path: str = typer.Option(".", "--path", "-p", help="Project path"),
):
    """Initialize a new Co-Scientist project."""
    project_dir = Path(path)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    for subdir in ["data", "data/vector_db", "data/hypotheses", "data/checkpoints", "data/results", "config", "domains"]:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create default config
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        default_config = """project_name: generic-discovery
max_iterations: 10

database:
  vector_db_path: "./data/vector_db"
  hypothesis_store_path: "./data/hypotheses"
  checkpoint_dir: "./data/checkpoints"
  results_dir: "./data/results"

domain:
  name: "generic"
  safety_module: "default_safety"
  validation_module: "default_validator"

agents:
  max_hypotheses_per_batch: 10
  similarity_threshold: 0.85
  min_debates_per_hypothesis: 10
"""
        config_path.write_text(default_config)

    # Create .env template
    env_path = project_dir / ".env.example"
    if not env_path.exists():
        env_template = """# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Search API Keys
SEMANTIC_SCHOLAR_API_KEY=your_ss_key
GOOGLE_SCHOLAR_SERPAPI_KEY=your_serpapi_key
"""
        env_path.write_text(env_template)

    console.print(f"[green]Initialized Co-Scientist project at: {project_dir.absolute()}[/green]")
    console.print(f"[dim]Config: {config_path}[/dim]")
    console.print(f"[dim]Env template: {env_path}[/dim]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Copy .env.example to .env and add your API keys")
    console.print("2. Edit config.yaml for your needs")
    console.print("3. Run: co-scientist research 'Your research goal'")


if __name__ == "__main__":
    app()
