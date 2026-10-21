"""Offline coverage of configuration, real SDK requests, and CLI behavior."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from agents import Model, OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai import APIConnectionError, AsyncOpenAI
from pydantic import ValidationError

from agent.agents.assistant import create_assistant
from agent.main import main, run_chat
from agent.models.settings import Settings
from agent.services.model_provider import configured_model
from agent.workflows.chat import ChatSession


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_openai_requires_key_without_exposing_input(self):
        with self.assertRaisesRegex(ValidationError, "OPENAI_API_KEY is required"):
            Settings(_env_file=None)
        settings = Settings(_env_file=None, openai_api_key=" secret-key ", openai_model="openai-test")
        self.assertEqual(settings.model_name, "openai-test")
        self.assertEqual(settings.openai_api_key.get_secret_value(), "secret-key")
        self.assertNotIn("secret-key", repr(settings))

    def test_ollama_requires_model_but_no_openai_key(self):
        with self.assertRaisesRegex(ValidationError, "OLLAMA_MODEL"):
            Settings(_env_file=None, model_provider="ollama")
        settings = Settings(_env_file=None, model_provider="ollama", ollama_model="gemma:test")
        self.assertEqual(settings.model_name, "gemma:test")
        self.assertIsNone(settings.openai_api_key)

    def test_invalid_provider_empty_models_and_url(self):
        for values in (
            {"model_provider": "unknown"},
            {"openai_api_key": " "},
            {"openai_api_key": "test", "openai_model": " "},
            {"model_provider": "ollama", "ollama_model": " "},
            {"model_provider": "ollama", "ollama_model": "test", "ollama_base_url": "bad-url"},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(_env_file=None, **values)

    def test_environment_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                "MODEL_PROVIDER=ollama\nOLLAMA_MODEL=file-model\nUNRELATED=value\n",
                encoding="utf-8",
            )
            os.environ["OLLAMA_MODEL"] = "env-model"
            settings = Settings(_env_file=dotenv)
        self.assertEqual(settings.model_name, "env-model")


class ProviderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_providers_replay_history_with_real_sdk(self):
        for provider in ("openai", "ollama"):
            with self.subTest(provider=provider), patch.dict(os.environ, {}, clear=True):
                requests = []

                def respond(request):
                    body = json.loads(request.content)
                    requests.append((request, body))
                    text = f"Reply {len(requests)}"
                    if provider == "ollama":
                        payload = {
                            "id": "chat-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "local-test",
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }],
                        }
                    else:
                        payload = {
                            "id": "resp-test",
                            "object": "response",
                            "created_at": 0,
                            "model": "openai-test",
                            "status": "completed",
                            "output": [{
                                "id": f"msg-{len(requests)}",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": text, "annotations": []}],
                            }],
                        }
                    return httpx.Response(200, json=payload)

                settings = Settings(
                    _env_file=None,
                    model_provider=provider,
                    openai_api_key="test-key",
                    openai_model="openai-test",
                    ollama_model="local-test",
                )
                http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                real_client = AsyncOpenAI

                def create_client(**kwargs):
                    self.assertEqual(
                        kwargs["base_url"],
                        "https://api.openai.com/v1" if provider == "openai"
                        else "http://localhost:11434/v1",
                    )
                    self.assertEqual(kwargs["api_key"], "test-key" if provider == "openai" else "ollama")
                    return real_client(**kwargs, http_client=http_client, max_retries=0)

                with patch("agent.services.model_provider.AsyncOpenAI", side_effect=create_client):
                    async with configured_model(settings) as model:
                        self.assertIsInstance(
                            model, OpenAIResponsesModel if provider == "openai"
                            else OpenAIChatCompletionsModel,
                        )
                        session = ChatSession(create_assistant(model))
                        self.assertTrue(session._run_config.tracing_disabled)
                        self.assertEqual(await session.reply("My product is milk."), "Reply 1")
                        self.assertEqual(await session.reply("Which product?"), "Reply 2")

                self.assertTrue(http_client.is_closed)
                self.assertEqual(len(requests), 2)
                expected_path = "/v1/responses" if provider == "openai" else "/v1/chat/completions"
                self.assertEqual(requests[0][0].url.path, expected_path)
                history = requests[1][1]["input" if provider == "openai" else "messages"]
                self.assertEqual([item["role"] for item in history if item["role"] != "system"],
                                 ["user", "assistant", "user"])
                self.assertIn("My product is milk.", json.dumps(history))
                self.assertIn("Reply 1", json.dumps(history))
                self.assertIn("Which product?", json.dumps(history))

    async def test_failed_turn_does_not_change_history(self):
        session = ChatSession(create_assistant(model=AsyncMock(spec=Model)))
        previous = [{"role": "user", "content": "Earlier question"}]
        session._history = previous.copy()
        with patch("agent.workflows.chat.Runner.run", new_callable=AsyncMock) as run:
            run.side_effect = APIConnectionError(request=httpx.Request("POST", "http://localhost"))
            with self.assertRaises(APIConnectionError):
                await session.reply("Failed question")
            self.assertEqual(session._history, previous)
            run.side_effect = None
            run.return_value.to_input_list = lambda: previous
            await session.reply("Retry question")
            self.assertEqual(run.call_args.kwargs["input"],
                             [*previous, {"role": "user", "content": "Retry question"}])


class CliTests(unittest.IsolatedAsyncioTestCase):
    async def test_blank_input_failure_recovery_and_exit(self):
        settings = Settings(_env_file=None, model_provider="ollama", ollama_model="test")
        error = APIConnectionError(request=httpx.Request("POST", "http://localhost"))
        with (
            patch("builtins.input", side_effect=[" ", "Hello", "Retry", "/exit"]),
            patch("agent.main.ChatSession.reply", new_callable=AsyncMock,
                  side_effect=[error, "Hello back"]) as reply,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            await run_chat(settings)
        self.assertEqual(reply.await_count, 2)
        self.assertIn("Connection failed", stderr.getvalue())
        self.assertIn("Assistant: Hello back", stdout.getvalue())
        self.assertIn("Goodbye!", stdout.getvalue())

    async def test_eof_exits_without_request(self):
        settings = Settings(_env_file=None, model_provider="ollama", ollama_model="test")
        with (
            patch("builtins.input", side_effect=EOFError),
            patch("agent.main.ChatSession.reply", new_callable=AsyncMock) as reply,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            await run_chat(settings)
        reply.assert_not_awaited()

    def test_configuration_error_exits_cleanly_without_secret(self):
        with (
            patch.dict(os.environ, {"MODEL_PROVIDER": "invalid", "OPENAI_API_KEY": "secret-key"}, clear=True),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaises(SystemExit) as caught,
        ):
            main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Invalid model_provider", stderr.getvalue())
        self.assertNotIn("secret-key", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
