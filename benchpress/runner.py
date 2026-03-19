from __future__ import annotations

import asyncio

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

from benchpress.client import BenchClient
from benchpress.config import BenchpressConfig, ProviderConfig, ModelConfig
from benchpress.models import PromptCase, RequestResult


async def run_model_benchmark(
    client: BenchClient,
    prompts: list[PromptCase],
    concurrency: int,
    interval_ms: int,
    max_tokens: int,
    temperature: float,
    progress: Progress,
    task_id: TaskID,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def run_one(prompt_case: PromptCase) -> None:
        async with sem:
            result = await client.send_request(
                prompt_case,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)
            progress.advance(task_id)

    tasks: list[asyncio.Task] = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(run_one(prompt)))
        if interval_ms > 0:
            await asyncio.sleep(interval_ms / 1000)

    await asyncio.gather(*tasks)
    return results


async def run_all_benchmarks(
    config: BenchpressConfig,
    prompts: list[PromptCase],
    verbose: bool = False,
) -> list[RequestResult]:
    all_results: list[RequestResult] = []

    # Count total tasks for progress bar
    total_tasks = sum(len(p.models) for p in config.providers) * len(prompts)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        for provider in config.providers:
            try:
                bearer_token = await provider.resolve_bearer_token()
            except ValueError as e:
                progress.console.print(f"[yellow]Warning: {e}, skipping provider '{provider.name}'[/yellow]")
                continue

            if provider.api_key_env.lower() != "none" and not bearer_token:
                progress.console.print(
                    f"[yellow]Warning: {provider.api_key_env} not set, "
                    f"skipping provider '{provider.name}'[/yellow]"
                )
                continue

            for model in provider.models:
                label = f"{model.display_name} ({provider.name})"
                task_id = progress.add_task(label, total=len(prompts))

                client = BenchClient(
                    base_url=provider.base_url,
                    api_key=bearer_token,
                    provider_name=provider.name,
                    model_id=model.id,
                    display_name=model.display_name,
                    timeout_s=config.test.timeout_s,
                    provider_extra_headers=provider.extra_headers,
                    model_extra_headers=model.extra_headers,
                )

                try:
                    results = await run_model_benchmark(
                        client=client,
                        prompts=prompts,
                        concurrency=config.test.concurrency,
                        interval_ms=config.test.interval_ms,
                        max_tokens=config.test.max_tokens,
                        temperature=config.test.temperature,
                        progress=progress,
                        task_id=task_id,
                    )
                    all_results.extend(results)

                    if verbose:
                        successes = sum(1 for r in results if r.status == "success")
                        avg_ms = (
                            sum(r.total_ms for r in results if r.status == "success")
                            / max(successes, 1)
                        )
                        progress.console.print(
                            f"  [green]{label}[/green]: {successes}/{len(results)} "
                            f"success, avg {avg_ms:.0f}ms"
                        )
                finally:
                    await client.close()

    return all_results
