"""Два транспорта с одним контрактом. Структуру ответов проверяем и после HTTP."""

from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar

import httpx
from pydantic import ValidationError

from .schema import Form

T = TypeVar("T", bound=Form)
DEFAULT_A = "qwen2.5:7b"
DEFAULT_B = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    def ask(self, model: str, system: str, prompt: str, form: type[T], *,
            choices: dict[str, list[str]] | None = None, budget: int = 500) -> T: ...


class Gateway:
    def __init__(self, http: httpx.Client, ollama_url: str, seed: int):
        self.http = http
        self.ollama_url = ollama_url.rstrip("/")
        self.seed = seed

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        try:
            response = self.http.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = "проверьте ключ и доступ к модели" if headers else exc.response.text[:400]
            if not headers and exc.response.status_code == 500 and "token repeat limit" in detail:
                # Ollama оборвала декодирование до получения JSON. Новый запрос с
                # короткой корректировкой проходит через тот же лимит, что и битый JSON.
                raise ValueError("генерация зациклилась, сформулируй короткий ответ другими словами") from exc
            raise GenerationError(f"{exc.response.status_code} от {url}: {detail}") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise GenerationError(f"Не получен ответ от {url}: {exc}") from exc

    def _complete(self, model: str, system: str, prompt: str, schema: dict,
                  name: str, budget: int) -> str:
        if model.startswith("openai/"):
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise GenerationError("Для выбранной модели openai/ нужен OPENAI_API_KEY.")
            body = self._post("https://api.openai.com/v1/responses", {
                "model": model.removeprefix("openai/"), "instructions": system,
                "input": prompt, "store": False, "max_output_tokens": max(2000, 3 * budget),
                "text": {"format": {"type": "json_schema", "name": name,
                                    "schema": schema, "strict": True}},
            }, {"Authorization": f"Bearer {key}"})
            if body.get("status") != "completed":
                raise ValueError(f"статус Responses API: {body.get('status')}")
            return "".join(part["text"] for item in body["output"] if item["type"] == "message"
                           for part in item["content"] if part["type"] == "output_text")
        # У GPT-OSS именованные уровни reasoning. В общий лимит Ollama входят
        # и эти токены, даже когда приложение читает только финальный JSON.
        is_gpt_oss = model.rsplit("/", 1)[-1].split(":")[0] == "gpt-oss"
        body = self._post(self.ollama_url + "/api/chat", {
            "model": model, "stream": False, "think": "low" if is_gpt_oss else False, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"seed": self.seed, "temperature": 0.65, "num_ctx": 8192,
                        "num_predict": budget + (1500 if is_gpt_oss else 0)},
        })
        if body.get("done_reason") == "length":
            raise ValueError("исчерпан лимит генерации")
        return body["message"]["content"]

    def ask(self, model: str, system: str, prompt: str, form: type[T], *,
            choices: dict[str, list[str]] | None = None, budget: int = 500) -> T:
        schema = form.model_json_schema()
        choices = choices or {}
        for field, allowed in choices.items():
            schema["properties"][field]["enum"] = allowed
        correction = ""
        for attempt in range(2):
            try:
                raw = self._complete(model, system, prompt + correction, schema,
                                     form.__name__, budget * (attempt + 1))
                result = form.model_validate_json(raw)
                for field, allowed in choices.items():
                    if getattr(result, field) not in allowed:
                        raise ValueError(f"{field}: допустимы {json.dumps(allowed, ensure_ascii=False)}")
                return result
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                if attempt == 1:
                    raise GenerationError(f"{model}: некорректный ответ для {form.__name__}: {exc}") from exc
                detail = ("; ".join(e["msg"] for e in exc.errors(include_input=False))
                          if isinstance(exc, ValidationError) else str(exc))
                correction = f"\nПерепиши ответ. {detail[:300]}. Верни объект JSON с текстом по-русски."
        raise AssertionError("unreachable")
