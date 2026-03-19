from __future__ import annotations

import re

import numpy as np

from benchpress.config import ModelConfig
from benchpress.models import (
    CategoryBreakdown,
    ExpectedToolCall,
    ModelSummary,
    PromptCase,
    RequestResult,
)


def score_tool_accuracy(
    result: RequestResult,
    expected_calls: list[ExpectedToolCall] | None,
    expected_no_tool: bool,
) -> float:
    if expected_no_tool:
        # Conversational: penalize any tool calls
        if result.tool_calls_made:
            return 0.0
        return 1.0

    if not expected_calls:
        return 1.0

    if not result.tool_calls_made:
        return 0.0

    # Score each expected call against the best-matching actual call
    actual_remaining = list(result.tool_calls_made)
    call_scores: list[float] = []

    for expected in expected_calls:
        best_score = 0.0
        best_idx = -1

        for i, actual in enumerate(actual_remaining):
            score = _score_single_call(expected, actual)
            if score > best_score:
                best_score = score
                best_idx = i

        call_scores.append(best_score)
        if best_idx >= 0:
            actual_remaining.pop(best_idx)

    # Penalize extra unexpected tool calls
    extra_penalty = len(actual_remaining) * 0.1
    return max(0.0, (sum(call_scores) / len(call_scores)) - extra_penalty)


def _score_single_call(expected: ExpectedToolCall, actual: dict) -> float:
    score = 0.0

    # Function name match (0.4)
    if actual.get("name") == expected.function_name:
        score += 0.4

    args = actual.get("arguments", {})

    # Required params present (0.3)
    expected_keys = set(expected.arguments.keys())
    actual_keys = set(args.keys())
    if expected_keys and expected_keys.issubset(actual_keys):
        score += 0.3

    # Param values match (0.3)
    if expected_keys:
        matches = 0
        for key in expected_keys:
            expected_val = expected.arguments[key]
            actual_val = args.get(key)
            if _values_match(expected_val, actual_val):
                matches += 1
        score += 0.3 * (matches / len(expected_keys))

    return score


def _values_match(expected, actual) -> bool:
    if expected == actual:
        return True
    # Handle numeric near-equality (for weight_lbs)
    try:
        return abs(float(expected) - float(actual)) < 0.01
    except (TypeError, ValueError):
        pass
    # Handle string comparison
    return str(expected).strip().lower() == str(actual).strip().lower()


def score_coherence(result: RequestResult) -> float:
    if result.status != "success":
        return 0.0

    score = 0.0
    text = result.response_text or ""
    has_tool_calls = bool(result.tool_calls_made)

    # Non-empty valid response (0.3)
    if text.strip() or has_tool_calls:
        score += 0.3

    # Well-structured (0.2): contains sentences with punctuation
    if text.strip():
        sentences = re.split(r"[.!?]+", text)
        real_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(real_sentences) >= 1:
            score += 0.2
    elif has_tool_calls:
        # Tool-only responses are fine
        score += 0.2

    # Proportional length (0.2): not absurdly short or long
    if text.strip():
        word_count = len(text.split())
        if 5 <= word_count <= 1000:
            score += 0.2
    elif has_tool_calls:
        score += 0.2

    # Valid tool call args (0.3)
    if has_tool_calls:
        valid_args = all(
            isinstance(tc.get("arguments"), dict) for tc in result.tool_calls_made
        )
        if valid_args:
            score += 0.3
    elif text.strip():
        # For text-only responses, award if the text is not garbled
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio > 0.5:
            score += 0.3

    return min(score, 1.0)


def score_relevance(result: RequestResult, prompt_text: str) -> float:
    if result.status != "success":
        return 0.0

    # Extract keywords from prompt (nouns, significant words)
    prompt_words = _extract_keywords(prompt_text)
    if not prompt_words:
        return 0.5  # neutral if no keywords

    # Check response text
    response_text = result.response_text or ""

    # Also include tool call info in response content
    if result.tool_calls_made:
        for tc in result.tool_calls_made:
            response_text += f" {tc.get('name', '')} "
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                response_text += " ".join(str(v) for v in args.values())

    response_words = _extract_keywords(response_text)
    if not response_words:
        return 0.0

    # Jaccard-like overlap
    overlap = prompt_words & response_words
    union = prompt_words | response_words
    return len(overlap) / len(union) if union else 0.0


def _extract_keywords(text: str) -> set[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such", "no",
        "than", "too", "very", "just", "also", "then", "that", "this",
        "these", "those", "it", "its", "i", "you", "he", "she", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "our", "their", "what", "which", "who", "whom", "if", "how",
        "when", "where", "why",
    }
    words = set(re.findall(r"[a-z0-9][\w-]*", text.lower()))
    return words - stop_words


def compute_cost(
    result: RequestResult,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> float | None:
    if result.input_tokens is None and result.output_tokens is None:
        return None
    input_cost = ((result.input_tokens or 0) / 1000) * cost_per_1k_input
    output_cost = ((result.output_tokens or 0) / 1000) * cost_per_1k_output
    return input_cost + output_cost


def score_results(
    results: list[RequestResult],
    prompt_cases: dict[str, PromptCase],
    model_configs: dict[tuple[str, str], ModelConfig],
) -> list[RequestResult]:
    """Score all results in-place and return them."""
    for result in results:
        pc = prompt_cases.get(result.prompt_case_id)
        if not pc:
            continue

        # Extract the user prompt text
        prompt_text = ""
        for msg in pc.messages:
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")

        result.tool_accuracy = score_tool_accuracy(
            result, pc.expected_tool_calls, pc.expected_no_tool
        )
        result.coherence = score_coherence(result)
        result.relevance = score_relevance(result, prompt_text)

        mc = model_configs.get((result.provider, result.model))
        if mc:
            result.cost_usd = compute_cost(
                result, mc.cost_per_1k_input, mc.cost_per_1k_output
            )

    return results


def compute_summaries(
    results: list[RequestResult],
    model_configs: dict[tuple[str, str], ModelConfig],
) -> list[ModelSummary]:
    """Group results by provider+model and compute aggregate stats."""
    groups: dict[tuple[str, str], list[RequestResult]] = {}
    for r in results:
        key = (r.provider, r.model)
        groups.setdefault(key, []).append(r)

    summaries: list[ModelSummary] = []
    for (provider, model), group in groups.items():
        mc = model_configs.get((provider, model))
        display_name = mc.display_name if mc else model

        successes = [r for r in group if r.status == "success"]
        errors = [r for r in group if r.status == "error"]
        timeouts = [r for r in group if r.status == "timeout"]

        ttfb_values = [r.ttfb_ms for r in successes if r.ttfb_ms is not None]
        total_values = [r.total_ms for r in successes]

        def percentile(vals: list[float], p: float) -> float:
            if not vals:
                return 0.0
            return float(np.percentile(vals, p))

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        # Per-category breakdowns
        categories: dict[str, list[RequestResult]] = {}
        for r in group:
            categories.setdefault(r.category, []).append(r)

        breakdowns = []
        for cat, cat_results in categories.items():
            cat_successes = [r for r in cat_results if r.status == "success"]
            breakdowns.append(
                CategoryBreakdown(
                    category=cat,
                    count=len(cat_results),
                    success_count=len(cat_successes),
                    avg_tool_accuracy=avg(
                        [r.tool_accuracy for r in cat_successes if r.tool_accuracy is not None]
                    ),
                    avg_coherence=avg(
                        [r.coherence for r in cat_successes if r.coherence is not None]
                    ),
                    avg_relevance=avg(
                        [r.relevance for r in cat_successes if r.relevance is not None]
                    ),
                    avg_total_ms=avg([r.total_ms for r in cat_successes]),
                )
            )

        total_input = sum(r.input_tokens or 0 for r in successes)
        total_output = sum(r.output_tokens or 0 for r in successes)
        total_cost = sum(r.cost_usd or 0.0 for r in successes)

        summaries.append(
            ModelSummary(
                provider=provider,
                model=model,
                display_name=display_name,
                total_requests=len(group),
                success_count=len(successes),
                error_count=len(errors),
                timeout_count=len(timeouts),
                success_rate=len(successes) / len(group) * 100 if group else 0.0,
                avg_ttfb_ms=avg(ttfb_values),
                p50_ttfb_ms=percentile(ttfb_values, 50),
                p95_ttfb_ms=percentile(ttfb_values, 95),
                avg_total_ms=avg(total_values),
                p50_total_ms=percentile(total_values, 50),
                p95_total_ms=percentile(total_values, 95),
                avg_tool_accuracy=avg(
                    [r.tool_accuracy for r in successes if r.tool_accuracy is not None]
                ),
                avg_coherence=avg(
                    [r.coherence for r in successes if r.coherence is not None]
                ),
                avg_relevance=avg(
                    [r.relevance for r in successes if r.relevance is not None]
                ),
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                avg_input_tokens=total_input / len(successes) if successes else 0.0,
                avg_output_tokens=total_output / len(successes) if successes else 0.0,
                total_cost_usd=total_cost,
                avg_cost_per_request=total_cost / len(successes) if successes else 0.0,
                category_breakdowns=breakdowns,
            )
        )

    return summaries
