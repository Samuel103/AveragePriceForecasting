"""Adapt provider configuration to an Agents SDK model."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from agents import Model, OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai import AsyncOpenAI

from agent.models.settings import Settings


@asynccontextmanager
async def configured_model(settings: Settings) -> AsyncIterator[Model]:
    if settings.model_provider == "ollama":
        client = AsyncOpenAI(
            base_url=str(settings.ollama_base_url),
            # The OpenAI client requires a value; local Ollama ignores it.
            api_key="ollama",
        )
        model_type = OpenAIChatCompletionsModel
    else:
        # Explicit endpoint keeps the selected provider independent of ambient
        # OPENAI_BASE_URL values in the shell.
        client = AsyncOpenAI(
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key.get_secret_value(),
        )
        model_type = OpenAIResponsesModel

    async with client:
        yield model_type(model=settings.model_name, openai_client=client)
