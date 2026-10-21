# Average price forecasting

## Command-line AI assistant

Install dependencies with `uv sync`, then copy `.env.example` to `.env` in the
repository root. Environment variables override values in `.env`; the file is
loaded from the directory where you start the app.

For OpenAI, set `MODEL_PROVIDER=openai`, supply `OPENAI_API_KEY`, and choose
`OPENAI_MODEL` (default: `gpt-4.1-mini`). This uses the OpenAI Responses API.

For local Ollama, start Ollama (`ollama serve` if it is not already running),
download your chosen model with `ollama pull <model-tag>`, and use the exact tag
shown by `ollama list`:

```dotenv
MODEL_PROVIDER=ollama
OLLAMA_MODEL=your-installed-model-tag
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Replace `your-installed-model-tag` with your installed model, including a Gemma
model if available in your Ollama installation. No OpenAI API key is needed in
Ollama mode. The client uses Ollama's OpenAI-compatible Chat Completions endpoint;
no API key needs to be configured for local Ollama.

Start the chat from the repository root:

```sh
uv run agent
# Alternative: uv run python -m agent
```

Use `/exit`, `/quit`, Ctrl-C, or Ctrl-D to leave. Empty input is ignored. Successful
turns are kept in memory until exit; failed requests leave that history intact.
History is not saved between launches. SDK tracing is disabled for both providers.
For questions about forecast results, the assistant is instructed to call its
read-only `read_forecasts` tool, which reads the current pipeline export at
`src/forecasting/data/predictions/forecasts.json` independently of the working
directory. The old `data/forecasts.json` is not used. The export is reread on each
query so newly generated predictions are available without restarting the chat.

The tool filters by full product name or case/accent-insensitive substring and
an exact month (`date`, `YYYY-MM`) or inclusive range (`start_date`, `end_date`).
Omitting filters lists available predictions and French product names. Product
variants and package sizes remain separate. It returns sorted prices and, when
at least two months match a product, the first-to-last absolute/percentage change
and trend across available observations. A zero starting price has no percentage
change. Missing months are not filled, and a single month has no trend.
Missing/unreadable files, malformed exports, unknown products and unavailable
months return explicit statuses instead of fabricated prices.

For example: “Quel est le prix prévu des bananes en octobre 2026 ?” or
“Comment évolue le prix prévu du lait entre septembre et octobre 2026 ?”
The assistant has no observed historical-price or web-research tools.
Ollama models must support function calling to query the forecasts.

The code follows `src/agent/AGENTS.md`: `agents/` defines the assistant,
`models/settings.py` validates configuration, `services/model_provider.py` selects
and manages the model client, `workflows/chat.py` runs conversations, and `main.py`
handles the CLI. `tools/read_forecasts.py` exposes the SDK tool and
`services/forecasts.py` reads, validates and queries the export.

Run the offline tests (no credentials or running Ollama needed):

```sh
uv run python -m unittest discover -s tests -v
```

Provider and history integration follow the official OpenAI documentation on
[models and providers](https://developers.openai.com/api/docs/guides/agents/models)
and [conversation results and state](https://developers.openai.com/api/docs/guides/agents/results).

## Forecasting pipeline

The `forecasting` package in `src/forecasting/` shares the root uv project and
Python environment with the agent. It extracts the original notebook, using
Polars for data preparation and the original LightGBM parameters for one model
across all products.

Run the full pipeline from the repository root:

```sh
uv run forecast
# Alternative: uv run python -m forecasting
```

The pipeline filters the source CSV to Québec from January 2024, builds monthly
features, evaluates a chronological holdout, refits on all eligible observations,
saves the model, and forecasts the next two consecutive months recursively.
Paths resolve from the forecasting component directory, independently of the
current working directory.

- Source CSV: `src/forecasting/data/raw/18100245.csv` (semicolon separated).
- Model: `src/forecasting/models/forecast_model.pkl`, including the product vocabulary.
- Forecasts: `src/forecasting/data/predictions/forecasts.json`, replacing the former
  repository-root `data/forecasts.json` export. Each record contains `PRODUCTS`,
  `DATE` (`YYYY-MM`), and numeric `VALUE`.

Each feature row describes a target month. `LAG_1`, `LAG_2`, and `LAG_3` use the
three preceding calendar months; `ROLLING_MEAN_3` averages them and
`PRICE_CHANGE_3 = LAG_1 - LAG_3`. Missing months and nonfinite prices remain missing.
Training requires an observed target and three complete preceding months.

The latest eligible observed month is held out for MAE (dollars) and WAPE
(percent); the evaluated model fits only earlier months. WAPE is N/A when total
actual prices are zero. A separate final model fits all eligible data. Predictions
retain full precision through both recursive steps. Incomplete recent history
for any product or nonfinite predictions abort forecasting before JSON export.

Run all tests (agent and forecasting) from the repository root:

```sh
uv run python -m unittest discover -s tests -v
```

To reuse a saved model:

```python
from forecasting.data_loader import load_prices
from forecasting.train import load_model
from forecasting.predict import forecast_two_months, export_forecasts

predictions = forecast_two_months(load_prices(), load_model())
export_forecasts(predictions)
```

Load only trusted model artifacts; persistence uses Python pickle.

`src/forecasting/notebooks/exploration.ipynb` and `training.ipynb` import the
production functions for experimentation. Install their optional dependencies
with `uv sync --group notebooks` from the repository root and select the root
`.venv` Python environment as the notebook kernel.

On macOS, install the OpenMP runtime with `brew install libomp` before running
LightGBM (see the [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)).
