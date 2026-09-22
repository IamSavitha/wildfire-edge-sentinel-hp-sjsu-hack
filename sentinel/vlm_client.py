"""OpenAI-compatible client for the local vLLM context classifier."""
import base64

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


class ContextVLM:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 timeout_s: float = 5.0, max_tokens: int = 160, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or OpenAI(base_url=base_url, api_key="EMPTY",
                                       timeout=timeout_s, max_retries=0)

    def _messages(self, jpeg: bytes) -> list[dict]:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]

    def classify(self, jpeg: bytes) -> tuple[ContextResult | None, int]:
        """Returns (context or None, total tokens spent)."""
        tokens = 0
        for _ in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=self._messages(jpeg),
                    max_tokens=self.max_tokens, temperature=0,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "context", "schema": CONTEXT_JSON_SCHEMA}},
                )
            except Exception:
                return None, tokens
            tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
            try:
                return ContextResult.model_validate_json(resp.choices[0].message.content), tokens
            except ValidationError:
                continue
        return None, tokens
