"""Command-line entry point; use `agent` or `python -m agent`."""

import asyncio
import sys

from agents.exceptions import AgentsException
from openai import APIConnectionError, APIStatusError, OpenAIError
from pydantic import ValidationError

from agent.agents.assistant import create_assistant
from agent.models.settings import Settings
from agent.services.model_provider import configured_model
from agent.workflows.chat import ChatSession


async def run_chat(settings: Settings) -> None:
    async with configured_model(settings) as model:
        session = ChatSession(create_assistant(model))
        print(f"Price forecasting assistant ({settings.model_provider}: {settings.model_name})")
        print("Type /exit or /quit to leave. History lasts for this session only.")
        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                return
            if message.lower() in {"/exit", "/quit", "exit", "quit"}:
                print("Goodbye!")
                return
            if not message:
                continue
            try:
                answer = await session.reply(message)
            except APIConnectionError:
                print(
                    "Connection failed. Check your network or that Ollama is running "
                    "and OLLAMA_BASE_URL is correct.",
                    file=sys.stderr,
                )
            except APIStatusError as error:
                print(
                    f"Provider returned HTTP {error.status_code}. Check your API key, "
                    "model name, and provider availability.",
                    file=sys.stderr,
                )
            except (OpenAIError, AgentsException):
                print("The agent could not complete this turn. Please try again.", file=sys.stderr)
            else:
                print(f"Assistant: {answer}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        # Report actionable fields without echoing input values or credentials.
        for issue in error.errors(include_input=False):
            field = ".".join(str(part) for part in issue["loc"]) or "configuration"
            print(f"Invalid {field}: {issue['msg']}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        asyncio.run(run_chat(settings))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
