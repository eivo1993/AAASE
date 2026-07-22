"""Thin wrapper around an LLM served through OpenRouter (OpenAI-compatible API)."""

import types
import typing
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config import settings

T = TypeVar("T", bound=BaseModel)


def _describe_type(annotation) -> str:
    """Human-readable shape of a type, expanding nested Pydantic models and lists
    so smaller models understand e.g. a list of {query, language} objects,
    not just the words 'list[SearchQuery]'."""
    origin = typing.get_origin(annotation)

    if origin is list:
        (item_type,) = typing.get_args(annotation)
        return f"list of {_describe_type(item_type)}"

    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _describe_type(args[0])

    if origin is typing.Literal:
        return " or ".join(repr(a) for a in typing.get_args(annotation))

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields = ", ".join(
            f'"{name}": {_describe_type(f.annotation)}'
            for name, f in annotation.model_fields.items()
        )
        return "{" + fields + "}"

    return getattr(annotation, "__name__", str(annotation))


def _describe_fields(schema: type[BaseModel]) -> str:
    """Flat 'field_name: type' listing. Simpler for smaller models to follow than a
    raw JSON Schema dump, which some models mistake for the answer to echo back."""
    lines = [
        f'- "{name}": {_describe_type(field.annotation)}'
        for name, field in schema.model_fields.items()
    ]
    return "\n".join(lines)


class LLMProvider:
    def __init__(self):
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        self._model = settings.openrouter_model

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Plain text completion. Returns the raw text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content

    def complete_structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        """Completion forced into JSON matching `schema`, parsed into that Pydantic model."""
        schema_instructions = (
            "Respond with ONLY a valid JSON object (no markdown, no extra text) containing "
            "exactly these fields, filled in with your actual answer (not the type names):\n"
            f"{_describe_fields(schema)}"
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": f"{system}\n\n{schema_instructions}"})
        else:
            messages.append({"role": "system", "content": schema_instructions})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        return schema.model_validate_json(content)
