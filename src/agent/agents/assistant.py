from agents import Agent, Model


def create_assistant(model: Model) -> Agent:
    return Agent(
        name="Price forecasting assistant",
        instructions=(
            "You are an assistant for an agri-food monthly price forecasting project. "
            "Respond in the user's language. Help explain prices and forecasting concepts. "
            "You currently have no access to historical prices, forecast results, files, "
            "web research, or external tools. Clearly explain these limits when relevant. "
            "Do not invent prices, forecasts, sources, or claim to have validated results."
        ),
        model=model,
    )
