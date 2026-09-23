"""OpenAI-compatible client for the context classifier: the local vLLM server on the edge, or the
same model (Qwen2.5-VL-7B-Instruct) behind a hosted OpenAI-compatible provider for the cloud-only
baseline (`cloud_vlm_from_env`). API keys are read from the environment only and never logged."""
import base64
import hashlib
import json
import logging
import os
import re
from typing import Callable, Mapping
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from pydantic import ValidationError

from sentinel.schema import CONTEXT_JSON_SCHEMA, ContextResult

SYSTEM_PROMPT = (
    "You are a wildfire lookout assistant. You see a cropped region from a remote tower "
    "camera where a detector flagged possible smoke or fire. Classify the source and its "
    "context. Benign sources (campfire, BBQ/chimney, industrial stack, controlled burn) and "
    "look-alikes (fog, dust, low cloud) are common, so look for fire rings, people, "
    "buildings, roads and smoke color. Answer ONLY with JSON matching the schema. "
    "description: one factual sentence, at most 25 words."
)
USER_PROMPT = "Classify this scene."

# Cloud-only baseline (scripts/bench_cloud.py): the camera's WHOLE frame, not a detector crop, so the
# model must also say when there is nothing there. smoke_color "none" is mapped to IGNORE.
CLOUD_FULL_FRAME_PROMPT = (
    "You are a wildfire lookout assistant. You see the FULL FRAME from a fixed lookout camera on a "
    "remote tower. Most frames contain no smoke at all. If there is no smoke or fire, answer "
    "smoke_color \"none\" and source_type \"fog_dust_cloud\". If there is smoke, classify its source "
    "and context. Benign sources (campfire, BBQ/chimney, industrial stack, controlled burn) and "
    "look-alikes (fog, dust, low cloud) are common, so look for fire rings, people, buildings, roads "
    "and smoke color. Answer ONLY with JSON matching the schema. "
    "description: one factual sentence, at most 25 words."
)

# json_schema: server-side constrained decoding (vLLM, some providers).
# json_object: provider "JSON mode"; the schema goes into the system prompt.
# none: no response_format at all; the schema goes into the system prompt.
RESPONSE_FORMATS = ("json_schema", "json_object", "none")

log = logging.getLogger(__name__)


def schema_hint() -> str:
    """The context schema as prompt text (field names + allowed values), for providers that
    cannot enforce a JSON schema server-side."""
    lines = ["Return a single JSON object with exactly these fields and nothing else:"]
    for name, spec in CONTEXT_JSON_SCHEMA["properties"].items():
        if "enum" in spec:
            allowed = "one of " + ", ".join(json.dumps(v) for v in spec["enum"])
        elif spec.get("type") == "boolean":
            allowed = "true or false"
        elif spec.get("type") == "string":
            allowed = "string" + (f", at most {spec['maxLength']} characters" if "maxLength" in spec else "")
        else:
            allowed = spec.get("type", "value")
        lines.append(f"- {name}: {allowed}")
    return "\n".join(lines)


def extract_json(text: str | None) -> dict | None:
    """First decodable {...} object in text (tolerates ```json fences and prose around it)."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def _redactor(secret: str | None) -> Callable[[str], str]:
    """Scrubs the key (and anything shaped like a bearer key) from text bound for logs/results."""
    def redact(text: str) -> str:
        if secret:
            text = text.replace(secret, "***")
        return re.sub(r"(sk-|Bearer\s+)[A-Za-z0-9_\-\.\*]+", r"\1***", text)
    return redact


def _openai_client(base_url: str, timeout_s: float, api_key: str | None = None) -> OpenAI:
    """HTTP URL, or unix:///path/to.sock for a zrt backend socket.

    The zrt proxy routes only by the served label, so a LoRA adapter name (e.g. "context")
    must be requested on the backend's own socket: unix:///opt/hp/zrt/run/vllm-<label>.sock
    """
    if base_url.startswith("unix://"):
        transport = httpx.HTTPTransport(uds=base_url[len("unix://"):])
        return OpenAI(base_url="http://localhost/v1", api_key="EMPTY", max_retries=0,
                      http_client=httpx.Client(transport=transport, timeout=timeout_s))
    return OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=0)


class ContextVLM:
    """parse_retries: extra requests after an unparseable reply (1 for the local server; 0 for a paid
    cloud API, where a retry at temperature 0 bills twice for the same answer)."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 timeout_s: float = 5.0, max_tokens: int = 160, client=None,
                 api_key: str | None = None, response_format: str = "json_schema",
                 system_prompt: str = SYSTEM_PROMPT, user_prompt: str = USER_PROMPT,
                 parse_retries: int = 1):
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}, got {response_format!r}")
        self.model = model
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.parse_retries = max(0, int(parse_retries))
        self.last_exception: Exception | None = None  # the latest request error itself (for retry policy)
        self.host = urlparse(base_url).hostname or base_url.split("://")[0]
        self.last_usage: dict = {}  # prompt/completion split of the latest classify() call
        self.last_error: str | None = None  # redacted request error of the latest call, if any
        self._redact = _redactor(api_key)
        self.client = client or _openai_client(base_url, timeout_s, api_key)

    def __repr__(self) -> str:
        return f"ContextVLM(model={self.model!r}, host={self.host!r}, response_format={self.response_format!r})"

    __str__ = __repr__

    def fingerprint(self) -> dict:
        """Everything that changes the answer to the same image: used as the response-cache key."""
        prompt = "\0".join((self.system_prompt, self.user_prompt, schema_hint()))
        return {"host": self.host, "model": self.model, "mode": self.response_format,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "max_tokens": self.max_tokens}

    def _messages(self, jpeg: bytes) -> list[dict]:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        system = self.system_prompt
        if self.response_format != "json_schema":
            system += "\n\n" + schema_hint()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": self.user_prompt},
            ]},
        ]

    def _request_kwargs(self) -> dict:
        if self.response_format == "json_schema":
            return {"response_format": {"type": "json_schema",
                                        "json_schema": {"name": "context", "schema": CONTEXT_JSON_SCHEMA}}}
        if self.response_format == "json_object":
            return {"response_format": {"type": "json_object"}}
        return {}

    def _parse(self, content: str | None) -> ContextResult:
        if self.response_format == "json_schema":
            try:
                return ContextResult.model_validate_json(content)
            except (ValidationError, ValueError, TypeError):
                pass  # e.g. a provider that wraps even schema-constrained output in a code fence
        obj = extract_json(content)
        if obj is None:
            raise ValueError("no JSON object in the reply")
        return ContextResult.model_validate(obj)

    def classify(self, jpeg: bytes) -> tuple[ContextResult | None, int]:
        """Returns (context or None, total tokens spent). The in/out split is kept in last_usage."""
        tokens = 0
        usage = self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.last_error = self.last_exception = None
        for _ in range(1 + self.parse_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=self._messages(jpeg),
                    max_tokens=self.max_tokens, temperature=0, **self._request_kwargs(),
                )
            except Exception as exc:
                self.last_exception = exc
                self.last_error = self._redact(f"{type(exc).__name__}: {exc}")[:300]
                log.warning("VLM request failed: %s", self.last_error)
                return None, tokens
            usage["calls"] += 1
            if resp.usage is None:  # some providers omit usage: count 0 and flag it
                usage["usage_missing"] = usage.get("usage_missing", 0) + 1
            else:
                prompt, completion = resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0
                tokens += prompt + completion
                usage["prompt_tokens"] += prompt
                usage["completion_tokens"] += completion
            try:
                return self._parse(resp.choices[0].message.content), tokens
            except (ValidationError, ValueError, TypeError):
                continue
        return None, tokens


def cloud_vlm_from_env(prefix: str = "CLOUD_VLM", env: Mapping[str, str] | None = None,
                       timeout_s: float = 60.0, response_format: str | None = None,
                       client=None, system_prompt: str = SYSTEM_PROMPT, parse_retries: int = 0) -> ContextVLM:
    """The cloud-only baseline's client, configured only from environment variables:
    <prefix>_BASE_URL, <prefix>_MODEL, <prefix>_API_KEY and optional <prefix>_RESPONSE_FORMAT
    (json_schema | json_object | none; `response_format` overrides it). The key is never printed.
    No parse retry by default: at temperature 0 it would bill twice for the same answer."""
    env = os.environ if env is None else env
    values = {}
    for suffix in ("BASE_URL", "MODEL", "API_KEY"):
        name = f"{prefix}_{suffix}"
        if not (env.get(name) or "").strip():
            raise SystemExit(f"environment variable {name} is not set (needed for the cloud VLM baseline)")
        values[suffix] = env[name].strip()
    fmt = response_format or (env.get(f"{prefix}_RESPONSE_FORMAT") or "").strip() or "json_schema"
    if fmt not in RESPONSE_FORMATS:
        raise SystemExit(f"{prefix}_RESPONSE_FORMAT must be one of {RESPONSE_FORMATS}, got {fmt!r}")
    return ContextVLM(values["MODEL"], values["BASE_URL"], timeout_s=timeout_s, client=client,
                      api_key=values["API_KEY"], response_format=fmt, system_prompt=system_prompt,
                      parse_retries=parse_retries)


def measure_net_baseline_ms(vlm: ContextVLM, n: int = 5, clock=None) -> float | None:
    """Median wall time of n cheap authenticated requests (GET /models) to the provider: the network
    + HTTPS overhead already inside every measured cloud latency. Subtracted before a modelled link is
    added, so the machine's own link is not counted twice. None if the provider refuses the request."""
    import statistics
    import time
    clock = clock or time.perf_counter
    samples = []
    for _ in range(n):
        t0 = clock()
        try:
            vlm.client.models.list()
        except Exception as exc:  # noqa: BLE001 - optional measurement
            log.warning("network baseline not measured: %s", vlm._redact(f"{type(exc).__name__}: {exc}")[:200])
            return None
        samples.append((clock() - t0) * 1000)
    return float(statistics.median(samples))


def ensure_served(client, model: str, base_url: str) -> None:
    """Fail fast if `model` is not served: classify() swallows request errors, so a wrong id or a
    down server would otherwise produce a results file full of parse failures."""
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise SystemExit(f"cannot list models at {base_url} ({exc}); is the VLM server up?") from exc
    if model not in ids:
        raise SystemExit(f"model {model!r} not served at {base_url}; available: {ids}")
