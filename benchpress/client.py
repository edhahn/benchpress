from __future__ import annotations

import json
import time

import httpx

from benchpress.models import PromptCase, RequestResult


class BenchClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        provider_name: str,
        model_id: str,
        display_name: str,
        timeout_s: int = 120,
        provider_extra_headers: dict[str, str] | None = None,
        model_extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url
        self.provider_name = provider_name
        self.model_id = model_id
        self.display_name = display_name

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider_extra_headers:
            headers.update(provider_extra_headers)
        if model_extra_headers:
            headers.update(model_extra_headers)

        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def send_request(
        self,
        prompt_case: PromptCase,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> RequestResult:
        payload: dict = {
            "model": self.model_id,
            "messages": prompt_case.messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if prompt_case.tools:
            payload["tools"] = prompt_case.tools

        start = time.monotonic()
        try:
            response = await self.client.post(
                "/chat/completions",
                json=payload,
            )
            total_ms = (time.monotonic() - start) * 1000
            ttfb_ms = total_ms  # non-streaming: TTFB ~ total

            if response.status_code != 200:
                return RequestResult(
                    prompt_case_id=prompt_case.id,
                    category=prompt_case.category,
                    provider=self.provider_name,
                    model=self.model_id,
                    display_name=self.display_name,
                    status="error",
                    total_ms=total_ms,
                    ttfb_ms=ttfb_ms,
                    error_message=f"HTTP {response.status_code}: {response.text[:500]}",
                )

            data = response.json()
            return self._parse_response(prompt_case, data, ttfb_ms, total_ms)

        except httpx.TimeoutException:
            total_ms = (time.monotonic() - start) * 1000
            return RequestResult(
                prompt_case_id=prompt_case.id,
                category=prompt_case.category,
                provider=self.provider_name,
                model=self.model_id,
                display_name=self.display_name,
                status="timeout",
                total_ms=total_ms,
                error_message="Request timed out",
            )
        except Exception as e:
            total_ms = (time.monotonic() - start) * 1000
            return RequestResult(
                prompt_case_id=prompt_case.id,
                category=prompt_case.category,
                provider=self.provider_name,
                model=self.model_id,
                display_name=self.display_name,
                status="error",
                total_ms=total_ms,
                error_message=str(e)[:500],
            )

    def _parse_response(
        self,
        prompt_case: PromptCase,
        data: dict,
        ttfb_ms: float,
        total_ms: float,
    ) -> RequestResult:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        response_text = message.get("content")
        tool_calls_raw = message.get("tool_calls")
        tool_calls_parsed = None

        if tool_calls_raw:
            tool_calls_parsed = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                tool_calls_parsed.append(
                    {"name": fn.get("name", ""), "arguments": args}
                )

        return RequestResult(
            prompt_case_id=prompt_case.id,
            category=prompt_case.category,
            provider=self.provider_name,
            model=self.model_id,
            display_name=self.display_name,
            status="success",
            ttfb_ms=ttfb_ms,
            total_ms=total_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            response_text=response_text,
            tool_calls_made=tool_calls_parsed,
        )
