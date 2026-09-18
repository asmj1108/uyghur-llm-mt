"""
translator.py - Downstream LLM Translation Engine
"""

import os
from typing import Optional, Dict, Any, List
from openai import OpenAI, AsyncOpenAI


class LLMTranslator:
    """
    Translates Uyghur sentences to English using in-context linguistic prompts.
    Supports OpenRouter, OpenAI, and custom OpenAI-compatible API endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "openai/gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", base_url)
        self.default_model = os.environ.get("TRANSLATION_LLM_MODEL", default_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None

    @property
    def client(self) -> Optional[OpenAI]:
        if not self.api_key:
            return None
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    @property
    def async_client(self) -> Optional[AsyncOpenAI]:
        if not self.api_key:
            return None
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._async_client

    def translate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Execute synchronous translation using LLM API.
        """
        if not self.api_key:
            return "[API Key Not Set: Set OPENROUTER_API_KEY or OPENAI_API_KEY to execute live LLM translation]"

        target_model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = dict(
                model=target_model,
                messages=messages,
                temperature=self.temperature,
            )
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Translation Error: {str(e)}]"

    async def translate_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Execute asynchronous translation using LLM API.
        """
        if not self.api_key:
            return "[API Key Not Set: Set OPENROUTER_API_KEY or OPENAI_API_KEY to execute live LLM translation]"

        target_model = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = dict(
                model=target_model,
                messages=messages,
                temperature=self.temperature,
            )
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            response = await self.async_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Translation Error: {str(e)}]"
