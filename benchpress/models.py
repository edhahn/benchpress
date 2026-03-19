from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExpectedToolCall(BaseModel):
    function_name: str
    arguments: dict


class PromptCase(BaseModel):
    id: str
    category: Literal["direct_tool", "reasoning_tool", "conversational"]
    messages: list[dict]
    tools: list[dict]
    expected_tool_calls: list[ExpectedToolCall] | None = None
    expected_no_tool: bool = False


class RequestResult(BaseModel):
    prompt_case_id: str
    category: str
    provider: str
    model: str
    display_name: str
    status: Literal["success", "error", "timeout"]
    ttfb_ms: float | None = None
    total_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_text: str | None = None
    tool_calls_made: list[dict] | None = None
    error_message: str | None = None

    # Computed scores
    tool_accuracy: float | None = None
    coherence: float | None = None
    relevance: float | None = None
    cost_usd: float | None = None


class CategoryBreakdown(BaseModel):
    category: str
    count: int
    success_count: int
    avg_tool_accuracy: float
    avg_coherence: float
    avg_relevance: float
    avg_total_ms: float


class ModelSummary(BaseModel):
    provider: str
    model: str
    display_name: str
    total_requests: int
    success_count: int
    error_count: int
    timeout_count: int
    success_rate: float

    avg_ttfb_ms: float
    p50_ttfb_ms: float
    p95_ttfb_ms: float
    avg_total_ms: float
    p50_total_ms: float
    p95_total_ms: float

    avg_tool_accuracy: float
    avg_coherence: float
    avg_relevance: float

    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_cost_usd: float
    avg_cost_per_request: float

    category_breakdowns: list[CategoryBreakdown]


class BenchmarkRun(BaseModel):
    timestamp: str
    config_summary: dict
    results: list[RequestResult]
    summaries: list[ModelSummary]
