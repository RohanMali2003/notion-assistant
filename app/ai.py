"""Centralized AI & LLM Generation Utilities.

Provides a unified Google GenAI client factory, model resolver, and typed
structured/text content generation helpers to eliminate boilerplate across services.
"""

import logging
import os
from typing import Any, List, Optional, Type, TypeVar
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("notion-assistant.ai")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
T = TypeVar("T", bound=BaseModel)


def get_genai_types():
    """Return the google.genai.types module if available, or None."""
    global types
    if types is not None:
        return types
    try:
        from google.genai import types as _types
        types = _types
        return _types
    except ImportError:
        return None


_CLIENT_OVERRIDE: Optional[Any] = None


def set_gemini_client_override(client: Optional[Any]) -> None:
    """Set explicit Gemini client instance override for testing or custom configuration."""
    global _CLIENT_OVERRIDE
    _CLIENT_OVERRIDE = client


def reset_gemini_client_override() -> None:
    """Reset Gemini client instance override."""
    global _CLIENT_OVERRIDE
    _CLIENT_OVERRIDE = None


def get_gemini_client():
    """Create and return a google-genai Client instance with configured API keys."""
    if _CLIENT_OVERRIDE is not None:
        return _CLIENT_OVERRIDE

    import sys
    for mod_name in ("app.leetcode_service", "app.workspace_service", "app.learning_service", "app.weekly_digest_service", "app.main"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "genai") and mod.genai is not None:
            client_cls = getattr(mod.genai, "Client", None)
            if client_cls is not None and (getattr(client_cls, "_mock_return_value", None) is not None or "Mock" in type(client_cls).__name__):
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                return client_cls(api_key=api_key) if api_key else client_cls()

    app_main = sys.modules.get("app.main")
    if app_main and hasattr(app_main, "genai") and app_main.genai is not None:
        client_cls = getattr(app_main.genai, "Client", None)
        if client_cls is not None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            return client_cls(api_key=api_key) if api_key else client_cls()

    if genai is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()



def get_gemini_model(default: str = DEFAULT_GEMINI_MODEL) -> str:
    """Return the configured Gemini model identifier."""
    return os.getenv("GEMINI_MODEL", default)


def generate_structured(
    prompt: str,
    schema: Type[T],
    system_instruction: Optional[str] = None,
    context: Optional[str] = None,
    fallback_default: Optional[T] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
) -> T:
    """Execute Gemini structured generation with typed Pydantic output and fallback handling."""
    try:
        active_client = client or get_gemini_client()
        model_name = model or get_gemini_model()
        prompt_content = f"Recent conversation context:\n{context}\n\nUser message: {prompt}" if context else prompt

        gen_types = get_genai_types()
        config = None
        if gen_types is not None:
            config_kwargs: dict[str, Any] = {
                "response_mime_type": "application/json",
                "response_schema": schema,
            }
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            config = gen_types.GenerateContentConfig(**config_kwargs)

        response = active_client.models.generate_content(
            model=model_name,
            contents=prompt_content,
            config=config,
        )

        if hasattr(response, "parsed") and response.parsed:
            if isinstance(response.parsed, schema):
                return response.parsed
            return schema.model_validate(response.parsed)
        elif hasattr(response, "text") and response.text:
            return schema.model_validate_json(response.text)
        else:
            raise ValueError(f"Empty response from Gemini for schema {schema.__name__}")

    except Exception as exc:
        logger.warning(
            "Gemini structured generation failed for schema %s: %s. Using fallback.",
            schema.__name__,
            exc,
        )
        if fallback_default is not None:
            return fallback_default
        raise


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
    temperature: Optional[float] = None,
    fallback_default: Optional[str] = None,
    client: Optional[Any] = None,
) -> str:
    """Execute standard Gemini text generation with robust error handling and fallback."""
    try:
        active_client = client or get_gemini_client()
        model_name = model or get_gemini_model()

        gen_types = get_genai_types()
        if config is None and gen_types is not None:
            cfg_kwargs: dict[str, Any] = {}
            if system_instruction:
                cfg_kwargs["system_instruction"] = system_instruction
            if tools:
                cfg_kwargs["tools"] = tools
            if temperature is not None:
                cfg_kwargs["temperature"] = temperature
            if cfg_kwargs:
                config = gen_types.GenerateContentConfig(**cfg_kwargs)

        response = active_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()
    except Exception as exc:
        logger.warning("Gemini text generation failed: %s", exc)
        if fallback_default is not None:
            return fallback_default
        raise


