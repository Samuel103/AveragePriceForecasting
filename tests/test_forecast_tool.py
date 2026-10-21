"""Offline forecast queries and real SDK tool-call round trips."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai import AsyncOpenAI

from agent.agents.assistant import create_assistant
from agent.services.forecasts import query_forecasts
from agent.workflows.chat import ChatSession


class ForecastQueryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "forecasts.json"
        self.rows = [
            {"PRODUCTS": "Café, 300 grammes", "DATE": "2026-10", "VALUE": 12.0},
            {"PRODUCTS": "Lait, 1 litre", "DATE": "2026-09", "VALUE": 3.0},
            {"PRODUCTS": "Café, 300 grammes", "DATE": "2026-09", "VALUE": 10.0},
            {"PRODUCTS": "Lait, 1 litre", "DATE": "2026-10", "VALUE": 2.0},
        ]
        self.write(self.rows)

    def write(self, rows):
        self.path.write_text(json.dumps(rows), encoding="utf-8")

    def query(self, **filters):
        return query_forecasts(path=self.path, **filters)

    def test_normalized_product_and_sorted_changes(self):
        result = self.query(product=" CAFE ")
        self.assertEqual(result["status"], "ok")
        self.assertEqual([row["VALUE"] for row in result["forecasts"]], [10.0, 12.0])
        self.assertEqual(result["changes"], [{
            "PRODUCTS": "Café, 300 grammes", "start_date": "2026-09", "end_date": "2026-10",
            "start_value": 10.0, "end_value": 12.0,
            "absolute_change": 2.0, "percent_change": 20.0, "trend": "increasing",
        }])
        self.assertEqual(self.query(product="lait")["changes"][0]["trend"], "decreasing")

    def test_exact_month_and_inclusive_open_ranges(self):
        for filters in (
            {"date": "2026-10"}, {"start_date": "2026-10"},
            {"start_date": "2026-10", "end_date": "2026-10"},
        ):
            with self.subTest(filters=filters):
                result = self.query(**filters)
                self.assertEqual(len(result["forecasts"]), 2)
                self.assertTrue(all(row["DATE"] == "2026-10" for row in result["forecasts"]))
                self.assertEqual(result["changes"], [])
        self.assertEqual(len(self.query(end_date="2026-09")["forecasts"]), 2)

    def test_discovery_unknown_product_unavailable_month_and_empty_export(self):
        self.assertEqual(len(self.query()["forecasts"]), 4)
        result = self.query(product="unknown")
        self.assertEqual(result["status"], "unknown_product")
        self.assertIn("Café, 300 grammes", result["available_products"])
        result = self.query(product="cafe", date="2027-01")
        self.assertEqual(result["status"], "no_forecasts")
        self.assertEqual(result["available_dates"], ["2026-09", "2026-10"])
        self.write([])
        self.assertEqual(self.query()["status"], "no_forecasts")

    def test_invalid_filters(self):
        for filters in (
            {"date": "2026-13"}, {"date": "2026-1"}, {"date": "0000-01"},
            {"date": "2026-10-01"}, {"product": " "},
            {"start_date": "2026-10", "end_date": "2026-09"},
            {"date": "2026-10", "start_date": "2026-09"},
        ):
            with self.subTest(filters=filters):
                self.assertEqual(self.query(**filters)["status"], "invalid_query")

    def test_missing_unreadable_and_corrupt_exports(self):
        self.path.unlink()
        self.assertEqual(self.query()["status"], "missing_file")
        self.path.mkdir()
        self.assertEqual(self.query()["status"], "unreadable_file")
        self.path.rmdir()
        self.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(self.query()["status"], "invalid_data")
        self.path.write_bytes(b"\xff")
        self.assertEqual(self.query()["status"], "unreadable_file")

    def test_invalid_records_are_never_used_as_prices(self):
        for rows in (
            {}, [None], [{}], [self.rows[0], self.rows[0]],
            [{**self.rows[0], "DATE": "2026-99"}],
            *([{**self.rows[0], "VALUE": value}] for value in (True, "12", None, float("nan"), float("inf"))),
        ):
            with self.subTest(rows=rows):
                self.write(rows)
                result = self.query()
                self.assertEqual(result["status"], "invalid_data")
                self.assertEqual(result["forecasts"], [])

    def test_exact_product_precedes_substring_and_variants_stay_separate(self):
        self.write([*self.rows, {**self.rows[0], "PRODUCTS": "Café, 300 grammes bio"}])
        self.assertEqual(len(self.query(product="Café, 300 grammes")["matched_products"]), 1)
        self.assertEqual(len(self.query(product="cafe")["matched_products"]), 2)

    def test_zero_baseline_mixed_trend_and_missing_months(self):
        self.write([
            {**self.rows[0], "DATE": "2026-01", "VALUE": 0},
            {**self.rows[0], "DATE": "2026-03", "VALUE": 2},
            {**self.rows[0], "DATE": "2026-04", "VALUE": 1},
        ])
        result = self.query()
        self.assertEqual(len(result["forecasts"]), 3)
        self.assertEqual(result["changes"][0]["trend"], "mixed")
        self.assertIsNone(result["changes"][0]["percent_change"])
        self.assertEqual(result["changes"][0]["absolute_change"], 1)
        self.write([{**row, "VALUE": 2} for row in self.rows])
        self.assertTrue(all(change["trend"] == "stable" for change in self.query()["changes"]))

    def test_queries_reread_export_and_do_not_modify_it(self):
        original = self.path.read_bytes()
        self.query()
        self.assertEqual(self.path.read_bytes(), original)
        self.write([{**self.rows[0], "VALUE": 42}])
        self.assertEqual(self.query()["forecasts"][0]["VALUE"], 42)


class ForecastToolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_executes_tool_and_returns_data_to_both_providers(self):
        for provider in ("openai", "ollama"):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "forecasts.json"
                path.write_text(json.dumps([
                    {"PRODUCTS": "Lait, 1 litre", "DATE": "2026-10", "VALUE": 2.5},
                ]), encoding="utf-8")
                requests = []
                arguments = json.dumps({"product": "lait", "date": "2026-10", "start_date": None, "end_date": None})

                def respond(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    first = len(requests) == 1
                    if provider == "ollama":
                        message = {"role": "assistant", "content": None if first else "2.50"}
                        if first:
                            message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {
                                "name": "read_forecasts", "arguments": arguments,
                            }}]
                        payload = {
                            "id": "chat-test", "object": "chat.completion", "created": 0, "model": "test",
                            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if first else "stop"}],
                        }
                    else:
                        output = ({"id": "fc-1", "type": "function_call", "call_id": "call-1",
                                   "name": "read_forecasts", "arguments": arguments, "status": "completed"}
                                  if first else {"id": "msg-1", "type": "message", "role": "assistant",
                                                 "status": "completed", "content": [{"type": "output_text", "text": "2.50", "annotations": []}]})
                        payload = {"id": "resp-test", "object": "response", "created_at": 0,
                                   "model": "test", "status": "completed", "output": [output]}
                    return httpx.Response(200, json=payload)

                async with (
                    httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client,
                    AsyncOpenAI(api_key="test", http_client=http_client) as client,
                ):
                    model_type = OpenAIResponsesModel if provider == "openai" else OpenAIChatCompletionsModel
                    session = ChatSession(create_assistant(model_type(model="test", openai_client=client)))
                    with patch("agent.services.forecasts.FORECASTS_PATH", path):
                        self.assertEqual(await session.reply("Prix prévu du lait en octobre 2026?"), "2.50")

                self.assertEqual(len(requests), 2)
                tool = requests[0]["tools"][0]
                definition = tool if provider == "openai" else tool["function"]
                self.assertEqual(definition["name"], "read_forecasts")
                self.assertEqual(set(definition["parameters"]["properties"]),
                                 {"product", "date", "start_date", "end_date"})
                history = requests[1]["input" if provider == "openai" else "messages"]
                outputs = [item["output"] if provider == "openai" else item["content"]
                           for item in history if item.get("type") == "function_call_output" or item.get("role") == "tool"]
                result = json.loads(outputs[0])
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["forecasts"][0]["VALUE"], 2.5)


if __name__ == "__main__":
    unittest.main()
