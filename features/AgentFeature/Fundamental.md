# Goal

Create the foundation of an AI agent application that will eventually be used to answer questions about forecasting results and validate those results using web research.

The main objective is to establish a clean, modular, and scalable project structure following the architecture defined in the `AGENTS.md` file.

# Acceptance Criteria

- Use the **OpenAI Agents SDK** to create and run the agent.
- Follow the architecture and project structure defined in `AGENTS.md`.
- Support both **OpenAI models** and **local Ollama models** (e.g., Gemma 4).
- Allow switching between OpenAI and Ollama through configuration without modifying the agent's implementation.
- Provide a simple command-line chat interface to interact with the agent.
- Support multi-turn conversations by preserving conversation history during the chat session.
- Separate agent definitions, model configuration, and application entry points to ensure maintainability and extensibility.
- Manage configuration and API keys using environment variables.

# Out of Scope

- Creating or implementing custom tools.
- Integrating MCP servers.
- Implementing web searches or forecasting data validation.
- Connecting the agent to forecasting results.
- Building a graphical user interface.