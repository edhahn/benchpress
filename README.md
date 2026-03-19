# Benchpress

CLI tool for benchmarking LLM providers and models. Measures response time, quality (tool accuracy, coherence, relevance), and cost — then generates a presentation-grade PDF report with LLM-generated insights.

Designed for comparing the same model across different providers (e.g., Claude Sonnet via Anthropic direct vs. via Gloo AI) to answer: is the provider adding latency, reducing quality, or saving enough money to justify the tradeoff?

## Features

- **Multi-provider benchmarking** — test multiple providers in a single run, each with its own auth (API key or OAuth client_credentials)
- **Head-to-head comparisons** — tag models with `base_model` to group the same underlying model across providers
- **LLM-generated insights** — configurable report LLM generates executive summaries and per-comparison analysis
- **Per-provider framing** — `insights_context` lets you control the tone of generated insights (e.g., acknowledge a provider's guardrails or value-add)
- **PDF reports** — timestamped, presentation-ready reports with side-by-side metrics and delta indicators
- **Flexible auth** — API keys via env vars, or OAuth2 client_credentials flow with Basic or POST body methods
- **Custom headers** — per-provider and per-model `extra_headers` for toggling features like extended context

## Installation

Requires Python 3.11+.

```bash
# Clone the repo
git clone <repo-url> && cd ai-benchpress

# Install in editable mode
pip install -e .

# Or with pipx for isolated install
pipx install -e .
```

## Quick Start

1. Copy the example config and environment file:

```bash
cp benchpress.yaml.example benchpress.yaml
cp .env.example .env
```

2. Add your API keys to `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
# For OAuth providers:
# GLOO_AI_CLIENT_ID=...
# GLOO_AI_CLIENT_SECRET=...
```

3. Edit `benchpress.yaml` to configure your providers and models.

4. Validate your config:

```bash
benchpress validate
```

5. Run benchmarks:

```bash
benchpress run -v
```

## Configuration

All configuration lives in `benchpress.yaml`. See `benchpress.yaml.example` for a fully documented example.

### Providers

Each provider needs a `name`, `base_url`, and authentication:

```yaml
providers:
  # API key auth
  - name: "anthropic"
    base_url: "https://api.anthropic.com/v1"
    api_key_env: "ANTHROPIC_API_KEY"
    models: [...]

  # OAuth client_credentials auth
  - name: "gloo-ai"
    base_url: "https://platform.ai.gloo.com/ai/v2/"
    oauth:
      token_url: "https://platform.ai.gloo.com/oauth2/token"
      client_id_env: "GLOO_AI_CLIENT_ID"
      client_secret_env: "GLOO_AI_CLIENT_SECRET"
      scopes: ["api/access"]
      auth_method: "basic"    # or "post_body"
    models: [...]

  # No auth (e.g., local Ollama)
  - name: "ollama"
    base_url: "http://localhost:11434/v1"
    models: [...]
```

### Cross-Provider Comparisons

Tag models with `base_model` to group the same underlying model across providers:

```yaml
# Under anthropic provider
- id: "claude-sonnet-4-5"
  display_name: "Sonnet 4.5"
  base_model: "claude-sonnet-4-5"

# Under gloo-ai provider
- id: "gloo-anthropic-claude-sonnet-4.5"
  display_name: "Gloo AI - Sonnet 4.5"
  base_model: "claude-sonnet-4-5"    # same tag = head-to-head comparison
```

### Provider Insights Context

Control how the report LLM frames each provider:

```yaml
- name: "gloo-ai"
  insights_context: |
    Gloo AI provides faith-based guardrails and prompt enhancements.
    Acknowledge the value these safety features add beyond raw performance.
```

### Extra Headers

Add custom headers at the provider or model level:

```yaml
- name: "anthropic"
  extra_headers:
    X-Custom-Header: "value"        # applied to all models
  models:
    - id: "claude-opus-4-6"
      extra_headers:
        anthropic-beta: "interleaved-thinking-2025-05-14"  # per-model
```

Model headers merge on top of provider headers (model wins on conflict).

### Report & Insights LLM

Configure which LLM generates report insights:

```yaml
report:
  results_dir: "./results"
  report_path: "./reports"

  # Option A: reference an existing provider
  llm_provider: "anthropic"
  llm_model: "claude-sonnet-4-5"

  # Option B: standalone config (mutually exclusive with Option A)
  # llm:
  #   base_url: "https://api.anthropic.com/v1"
  #   api_key_env: "ANTHROPIC_API_KEY"
  #   model: "claude-sonnet-4-5"
```

If no LLM is configured, reports are generated with metrics only.

## CLI Reference

```bash
# Run benchmarks
benchpress run [OPTIONS]
  -c, --config        Config file path (default: benchpress.yaml)
  -e, --env           .env file path
  -m, --models        Comma-separated model filter
  -p, --providers     Comma-separated provider filter
  -n, --num-requests  Override number of requests per model
  --concurrency       Override concurrency level
  --interval          Override interval between requests (ms)
  -o, --output        Override report output directory
  --no-pdf            Skip PDF generation, only save JSON
  -v, --verbose       Print per-model results during run

# Re-generate PDF from previous JSON results
benchpress report -i results/benchpress-YYYYMMDD-HHMMSS.json
  -c, --config        Config file (needed for LLM insights)
  -o, --output        Override report output directory

# List configured models
benchpress list-models

# Validate config and check env vars
benchpress validate
```

## Output

Each run produces:
- **JSON results** in `results/benchpress-YYYYMMDD-HHMMSS.json` — raw data for all requests
- **PDF report** in `reports/benchpress-report-YYYYMMDD-HHMMSS.pdf` — presentation-ready report

### Report Structure

1. **Title & Executive Summary** — config, winner badges, LLM-generated narrative
2. **Head-to-Head Comparisons** — side-by-side metrics with deltas for matched models
3. **Additional Models** — unmatched models in a compact table
4. **Data Appendix** — full metrics table, per-category breakdown, error summary

## Project Structure

```
ai-benchpress/
  benchpress/
    cli.py          # CLI commands (typer)
    config.py       # YAML config loading + auth (API key, OAuth)
    client.py       # HTTP client for OpenAI-compatible APIs
    runner.py       # Async benchmark orchestration
    prompts.py      # Test prompt generation
    scorer.py       # Result scoring (tool accuracy, coherence, relevance)
    models.py       # Pydantic data models
    report.py       # PDF report generation + LLM insight calls
  benchpress.yaml.example
  .env.example
  pyproject.toml
```

## Contributing

1. Fork the repo and create a feature branch
2. Install in editable mode: `pip install -e .`
3. Make your changes
4. Test with a small run: `benchpress run -n 3 -v`
5. Verify config loading: `benchpress validate`
6. Submit a pull request

### Adding a New Provider

1. Add the provider block to `benchpress.yaml` with auth config
2. Add models with `base_model` tags if comparing against existing providers
3. Add `insights_context` to frame the provider in report narratives
4. Run `benchpress validate` to check credentials
5. Test with `benchpress run -p your-provider -n 3 -v`

### Guidelines

- All providers use the OpenAI-compatible `/chat/completions` endpoint
- Keep `benchpress.yaml.example` up to date with any new config fields
- Credential values belong in `.env`, never in YAML files — use `*_env` fields to reference env var names
