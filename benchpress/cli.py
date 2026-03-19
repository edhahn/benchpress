from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from benchpress.config import load_config, ModelConfig
from benchpress.models import BenchmarkRun
from benchpress.prompts import PromptGenerator
from benchpress.report import generate_report
from benchpress.runner import run_all_benchmarks
from benchpress.scorer import compute_summaries, score_results

app = typer.Typer(
    name="benchpress",
    help="Benchmark LLM providers and models for response time, quality, and cost.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: str = typer.Option("benchpress.yaml", "--config", "-c", help="Config file path"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help=".env file path"),
    models: Optional[str] = typer.Option(None, "--models", "-m", help="Comma-separated model filter"),
    providers: Optional[str] = typer.Option(None, "--providers", "-p", help="Comma-separated provider filter"),
    num_requests: Optional[int] = typer.Option(None, "--num-requests", "-n", help="Override num_requests"),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Override concurrency"),
    interval: Optional[int] = typer.Option(None, "--interval", help="Override interval_ms"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Override report output directory"),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF, only save JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print each result"),
) -> None:
    """Run benchmarks against configured providers and models."""
    cfg = load_config(config, env)

    # Apply CLI overrides
    if num_requests is not None:
        cfg.test.num_requests = num_requests
    if concurrency is not None:
        cfg.test.concurrency = concurrency
    if interval is not None:
        cfg.test.interval_ms = interval

    # Filter providers/models
    if providers:
        provider_set = {p.strip() for p in providers.split(",")}
        cfg.providers = [p for p in cfg.providers if p.name in provider_set]
    if models:
        model_set = {m.strip() for m in models.split(",")}
        for p in cfg.providers:
            p.models = [m for m in p.models if m.id in model_set]
        cfg.providers = [p for p in cfg.providers if p.models]

    if not cfg.providers:
        console.print("[red]No providers/models to test after filtering.[/red]")
        raise typer.Exit(1)

    # Generate prompts
    gen = PromptGenerator(seed=cfg.test.seed)
    prompts = gen.generate(cfg.test.num_requests)

    console.print(f"\n[bold]Benchpress[/bold] - Testing {sum(len(p.models) for p in cfg.providers)} "
                  f"model(s) with {len(prompts)} prompts each\n")

    # Run benchmarks
    results = asyncio.run(run_all_benchmarks(cfg, prompts, verbose=verbose))

    # Build lookup dicts for scoring
    prompt_lookup = {p.id: p for p in prompts}
    model_lookup: dict[tuple[str, str], ModelConfig] = {}
    for p in cfg.providers:
        for m in p.models:
            model_lookup[(p.name, m.id)] = m

    # Score results
    score_results(results, prompt_lookup, model_lookup)
    summaries = compute_summaries(results, model_lookup)

    # Build run object
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    bench_run = BenchmarkRun(
        timestamp=timestamp,
        config_summary={
            "num_requests": cfg.test.num_requests,
            "concurrency": cfg.test.concurrency,
            "interval_ms": cfg.test.interval_ms,
            "temperature": cfg.test.temperature,
            "max_tokens": cfg.test.max_tokens,
            "seed": cfg.test.seed,
        },
        results=results,
        summaries=summaries,
    )

    # Save JSON results
    results_dir = Path(cfg.report.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"benchpress-{ts_slug}.json"
    with open(json_path, "w") as f:
        json.dump(bench_run.model_dump(), f, indent=2, default=str)
    console.print(f"\n[green]Results saved:[/green] {json_path}")

    # Generate PDF
    if not no_pdf:
        report_dir = Path(output or cfg.report.report_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"benchpress-report-{ts_slug}.pdf"
        asyncio.run(generate_report(bench_run, cfg, str(report_path)))
        console.print(f"[green]Report saved:[/green] {report_path}")

    # Print summary table
    _print_summary_table(summaries, cfg)


@app.command()
def report(
    config: str = typer.Option("benchpress.yaml", "--config", "-c", help="Config file path"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help=".env file path"),
    input: str = typer.Option(..., "--input", "-i", help="Path to JSON results file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="PDF output directory"),
) -> None:
    """Re-generate a PDF report from a previous run's JSON results."""
    path = Path(input)
    if not path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        raise typer.Exit(1)

    cfg = load_config(config, env)

    with open(path) as f:
        data = json.load(f)

    bench_run = BenchmarkRun(**data)

    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_dir = Path(output or cfg.report.report_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"benchpress-report-{ts_slug}.pdf"

    asyncio.run(generate_report(bench_run, cfg, str(report_path)))
    console.print(f"[green]Report saved:[/green] {report_path}")


@app.command("list-models")
def list_models(
    config: str = typer.Option("benchpress.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """List all configured providers and models."""
    cfg = load_config(config)

    table = Table(title="Configured Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Display Name")
    table.add_column("Base Model", style="dim")
    table.add_column("Cost (in/out per 1k)", style="yellow")

    for p in cfg.providers:
        for m in p.models:
            table.add_row(
                p.name,
                m.id,
                m.display_name,
                m.base_model or "-",
                f"${m.cost_per_1k_input:.4f} / ${m.cost_per_1k_output:.4f}",
            )

    console.print(table)


@app.command()
def validate(
    config: str = typer.Option("benchpress.yaml", "--config", "-c", help="Config file path"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help=".env file path"),
) -> None:
    """Validate config and check API key environment variables."""
    try:
        cfg = load_config(config, env)
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Config loaded:[/green] {config}")
    console.print(f"  Providers: {len(cfg.providers)}")
    console.print(f"  Total models: {sum(len(p.models) for p in cfg.providers)}")
    console.print(f"  Requests per model: {cfg.test.num_requests}")

    all_ok = True
    for p in cfg.providers:
        if p.oauth:
            oauth = p.oauth
            client_id = oauth.resolve_client_id()
            client_secret = oauth.resolve_client_secret()
            missing = []
            if not client_id:
                missing.append(oauth.client_id_env)
            if not client_secret:
                missing.append(oauth.client_secret_env)
            if missing:
                console.print(f"  [red]Missing: {', '.join(missing)} for provider '{p.name}'[/red]")
                all_ok = False
            else:
                console.print(
                    f"  [green]{p.name}:[/green] {p.base_url} "
                    f"(oauth/{oauth.auth_method} via {oauth.token_url})"
                )
        else:
            key = p.resolve_api_key()
            if p.api_key_env.lower() != "none" and not key:
                console.print(f"  [red]Missing: {p.api_key_env} for provider '{p.name}'[/red]")
                all_ok = False
            else:
                status = "no auth" if p.api_key_env.lower() == "none" else "api_key set"
                console.print(f"  [green]{p.name}:[/green] {p.base_url} ({status})")

    # Report LLM config
    report = cfg.report
    if report.llm_provider:
        console.print(f"  [green]Report LLM:[/green] {report.llm_provider} / {report.llm_model}")
    elif report.llm:
        console.print(f"  [green]Report LLM:[/green] {report.llm.base_url} / {report.llm.model}")
    else:
        console.print("  [yellow]Report LLM: not configured (insights disabled)[/yellow]")

    if all_ok:
        console.print("\n[green]Validation passed.[/green]")
    else:
        console.print("\n[yellow]Validation completed with warnings.[/yellow]")


def _print_summary_table(summaries: list, cfg=None) -> None:
    if not summaries:
        return

    table = Table(title="\nBenchmark Results Summary")
    if cfg:
        provider_names = ", ".join(p.name for p in cfg.providers)
        total_models = sum(len(p.models) for p in cfg.providers)
        table.caption = (
            f"Providers: {provider_names} ({total_models} models)\n"
            f"Requests/model: {cfg.test.num_requests} | "
            f"Concurrency: {cfg.test.concurrency} | "
            f"Interval: {cfg.test.interval_ms}ms | "
            f"Temperature: {cfg.test.temperature} | "
            f"Max tokens: {cfg.test.max_tokens} | "
            f"Seed: {cfg.test.seed}"
        )
        table.caption_justify = "left"

    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="dim")
    table.add_column("Success", justify="right")
    table.add_column("Avg ms", justify="right", style="yellow")
    table.add_column("P95 ms", justify="right", style="yellow")
    table.add_column("Avg In Tok", justify="right", style="dim")
    table.add_column("Avg Out Tok", justify="right", style="dim")
    table.add_column("Total Tok", justify="right", style="dim")
    table.add_column("Tool Acc", justify="right", style="green")
    table.add_column("Coherence", justify="right", style="green")
    table.add_column("Cost", justify="right", style="magenta")

    for s in summaries:
        table.add_row(
            s.display_name,
            s.provider,
            f"{s.success_rate:.0f}%",
            f"{s.avg_total_ms:.0f}",
            f"{s.p95_total_ms:.0f}",
            f"{s.avg_input_tokens:.0f}",
            f"{s.avg_output_tokens:.0f}",
            f"{s.total_input_tokens + s.total_output_tokens:,}",
            f"{s.avg_tool_accuracy:.0%}",
            f"{s.avg_coherence:.0%}",
            f"${s.total_cost_usd:.4f}",
        )

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
