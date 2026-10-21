from agents import Agent, RunConfig, Runner, TResponseInputItem


class ChatSession:
    """Keep successful conversation turns in memory for one chat session."""

    def __init__(self, assistant: Agent) -> None:
        self.assistant = assistant
        self._history: list[TResponseInputItem] = []
        # Avoid exporting conversation traces, including for fully local Ollama use.
        self._run_config = RunConfig(tracing_disabled=True)

    async def reply(self, message: str) -> str:
        pending: list[TResponseInputItem] = [
            *self._history,
            {"role": "user", "content": message},
        ]
        result = await Runner.run(
            self.assistant, input=pending, run_config=self._run_config
        )
        self._history = result.to_input_list()
        return str(result.final_output)
