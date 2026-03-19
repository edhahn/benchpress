from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Literal

import httpx
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator


class ModelConfig(BaseModel):
    id: str
    display_name: str
    base_model: str | None = None
    extra_headers: dict[str, str] | None = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class OAuthConfig(BaseModel):
    token_url: str
    client_id_env: str
    client_secret_env: str
    scopes: list[str] = []
    auth_method: Literal["basic", "post_body"] = "basic"

    def resolve_client_id(self) -> str | None:
        return os.environ.get(self.client_id_env)

    def resolve_client_secret(self) -> str | None:
        return os.environ.get(self.client_secret_env)


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str = "none"
    oauth: OAuthConfig | None = None
    insights_context: str | None = None
    extra_headers: dict[str, str] | None = None
    models: list[ModelConfig]

    @model_validator(mode="after")
    def check_auth_mutually_exclusive(self) -> ProviderConfig:
        has_api_key = self.api_key_env.lower() != "none"
        has_oauth = self.oauth is not None
        if has_api_key and has_oauth:
            raise ValueError(
                f"Provider '{self.name}': specify either api_key_env or oauth, not both"
            )
        return self

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolve_api_key(self) -> str | None:
        if self.api_key_env.lower() == "none":
            return None
        return os.environ.get(self.api_key_env)

    async def resolve_bearer_token(self) -> str | None:
        if self.oauth:
            return await self._exchange_oauth_token()
        return self.resolve_api_key()

    async def _exchange_oauth_token(self) -> str:
        oauth = self.oauth
        assert oauth is not None

        client_id = oauth.resolve_client_id()
        client_secret = oauth.resolve_client_secret()

        missing = []
        if not client_id:
            missing.append(oauth.client_id_env)
        if not client_secret:
            missing.append(oauth.client_secret_env)
        if missing:
            raise ValueError(
                f"Provider '{self.name}': missing OAuth env vars: {', '.join(missing)}"
            )

        data: dict[str, str] = {"grant_type": "client_credentials"}
        if oauth.scopes:
            data["scope"] = " ".join(oauth.scopes)

        headers: dict[str, str] = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        if oauth.auth_method == "basic":
            credentials = base64.b64encode(
                f"{client_id}:{client_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        else:
            data["client_id"] = client_id  # type: ignore[assignment]
            data["client_secret"] = client_secret  # type: ignore[assignment]

        async with httpx.AsyncClient() as http:
            resp = await http.post(oauth.token_url, data=data, headers=headers)
            if resp.status_code != 200:
                raise ValueError(
                    f"Provider '{self.name}': OAuth token exchange failed "
                    f"(HTTP {resp.status_code}): {resp.text[:500]}"
                )
            body = resp.json()

        access_token = body.get("access_token")
        if not access_token:
            raise ValueError(
                f"Provider '{self.name}': OAuth response missing 'access_token'"
            )
        return access_token


class TestConfig(BaseModel):
    num_requests: int = 20
    concurrency: int = 3
    interval_ms: int = 500
    timeout_s: int = 120
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int = 42

    @field_validator("num_requests", "concurrency")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("interval_ms")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v


class ReportLLMConfig(BaseModel):
    base_url: str
    api_key_env: str = "none"
    oauth: OAuthConfig | None = None
    model: str

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolve_api_key(self) -> str | None:
        if self.api_key_env.lower() == "none":
            return None
        return os.environ.get(self.api_key_env)

    async def resolve_bearer_token(self) -> str | None:
        if self.oauth:
            client_id = self.oauth.resolve_client_id()
            client_secret = self.oauth.resolve_client_secret()
            missing = []
            if not client_id:
                missing.append(self.oauth.client_id_env)
            if not client_secret:
                missing.append(self.oauth.client_secret_env)
            if missing:
                raise ValueError(
                    f"Report LLM: missing OAuth env vars: {', '.join(missing)}"
                )
            data: dict[str, str] = {"grant_type": "client_credentials"}
            if self.oauth.scopes:
                data["scope"] = " ".join(self.oauth.scopes)
            headers: dict[str, str] = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            if self.oauth.auth_method == "basic":
                credentials = base64.b64encode(
                    f"{client_id}:{client_secret}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
            else:
                data["client_id"] = client_id  # type: ignore[assignment]
                data["client_secret"] = client_secret  # type: ignore[assignment]
            async with httpx.AsyncClient() as http:
                resp = await http.post(self.oauth.token_url, data=data, headers=headers)
                resp.raise_for_status()
                return resp.json()["access_token"]
        return self.resolve_api_key()


class ReportConfig(BaseModel):
    results_dir: str = "./results"
    report_path: str = "./reports"
    llm_provider: str | None = None
    llm_model: str | None = None
    llm: ReportLLMConfig | None = None

    @model_validator(mode="after")
    def check_llm_mutually_exclusive(self) -> ReportConfig:
        has_ref = self.llm_provider is not None
        has_standalone = self.llm is not None
        if has_ref and has_standalone:
            raise ValueError(
                "report: specify either llm_provider/llm_model or llm block, not both"
            )
        if has_ref and self.llm_model is None:
            raise ValueError(
                "report: llm_model is required when llm_provider is set"
            )
        return self


class BenchpressConfig(BaseModel):
    providers: list[ProviderConfig]
    test: TestConfig = TestConfig()
    report: ReportConfig = ReportConfig()
    # Backward compat: accept 'output' key and map to report
    output: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_output_to_report(cls, values: dict) -> dict:
        if "output" in values and "report" not in values:
            values["report"] = values.pop("output")
        elif "output" in values and "report" in values:
            values.pop("output")
        return values


def load_config(
    config_path: str = "benchpress.yaml",
    env_path: str | None = None,
) -> BenchpressConfig:
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return BenchpressConfig(**raw)
