"""
Unified LLM client for OpenRouter and Google AI Studio.

Provides a common interface for sending summarization requests
to different model providers, with rate limiting and retry logic.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    model: str
    provider: str               # openrouter, google
    output: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    latency_seconds: float = 0.0


class LLMClient:
    """Unified client for OpenRouter and Google AI Studio."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        request_delay: float = 2.0,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay

        # Lazy-initialized clients
        self._openrouter_client = None
        self._google_client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarize(
        self,
        document: str,
        system_prompt: str,
        user_prompt_template: str,
        model_id: str,
        provider: str,
    ) -> LLMResponse:
        """
        Send a document for summarization to the specified model.

        Args:
            document: The document text (clean or poisoned).
            system_prompt: System instruction for the model.
            user_prompt_template: User prompt template with {document} placeholder.
            model_id: Model identifier (e.g., "meta-llama/llama-3.3-70b-instruct:free").
            provider: "openrouter" or "google".

        Returns:
            LLMResponse with the model's output.
        """
        user_prompt = user_prompt_template.format(document=document)

        if provider == "openrouter":
            return self._call_openrouter(system_prompt, user_prompt, model_id)
        elif provider == "google":
            return self._call_google(system_prompt, user_prompt, model_id)
        else:
            return LLMResponse(
                model=model_id,
                provider=provider,
                output="",
                error=f"Unknown provider: {provider}",
            )

    # ------------------------------------------------------------------
    # OpenRouter (OpenAI-compatible API)
    # ------------------------------------------------------------------

    def _get_openrouter_client(self):
        """Lazy-init the OpenRouter client."""
        if self._openrouter_client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable not set. "
                    "Get your key from https://openrouter.ai/settings/keys"
                )
            self._openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        return self._openrouter_client

    def _call_openrouter(
        self, system_prompt: str, user_prompt: str, model_id: str
    ) -> LLMResponse:
        """Call OpenRouter API with retry logic."""
        client = self._get_openrouter_client()

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = time.time() - start

                output = response.choices[0].message.content or ""
                usage = response.usage

                time.sleep(self.request_delay)

                return LLMResponse(
                    model=model_id,
                    provider="openrouter",
                    output=output.strip(),
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    latency_seconds=round(latency, 2),
                )

            except Exception as e:
                error_str = str(e)
                logger.warning(
                    f"OpenRouter attempt {attempt}/{self.max_retries} failed: {error_str}"
                )
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * attempt
                    # OpenRouter free tier (like Venice) sometimes has explicit 30s rate limits
                    if "429" in error_str:
                        sleep_time = max(sleep_time, 32.0)
                    logger.info(f"Sleeping for {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    return LLMResponse(
                        model=model_id,
                        provider="openrouter",
                        output="",
                        error=str(e),
                    )

    # ------------------------------------------------------------------
    # Google AI Studio (Gemini)
    # ------------------------------------------------------------------

    def _get_google_client(self):
        """Lazy-init the Google GenAI client."""
        if self._google_client is None:
            from google import genai

            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set. "
                    "Get your key from https://aistudio.google.com/apikey"
                )
            self._google_client = genai.Client(api_key=api_key)
        return self._google_client

    def _call_google(
        self, system_prompt: str, user_prompt: str, model_id: str
    ) -> LLMResponse:
        """Call Google AI Studio API with retry logic."""
        client = self._get_google_client()
        from google.genai import types

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                latency = time.time() - start

                output = response.text or ""

                time.sleep(self.request_delay)

                return LLMResponse(
                    model=model_id,
                    provider="google",
                    output=output.strip(),
                    latency_seconds=round(latency, 2),
                )

            except Exception as e:
                error_str = str(e)
                logger.warning(
                    f"Google attempt {attempt}/{self.max_retries} failed: {error_str}"
                )
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * attempt
                    if "429" in error_str:
                        sleep_time = max(sleep_time, 32.0)
                    logger.info(f"Sleeping for {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                else:
                    return LLMResponse(
                        model=model_id,
                        provider="google",
                        output="",
                        error=str(e),
                    )
