"""Environment-based configuration, independent of agent definitions."""

from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    model_provider: Literal["openai", "ollama"] = "openai"
    openai_model: str = Field(default="gpt-6-luna", min_length=1)
    openai_api_key: SecretStr | None = None
    ollama_model: str = Field(default="", description="Exact installed Ollama model tag")
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434/v1")

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.model_provider == "openai":
            if not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
                raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
        else:
            if not self.ollama_model:
                raise ValueError("OLLAMA_MODEL must match an installed Ollama model tag")
        return self

    @property
    def model_name(self) -> str:
        return self.openai_model if self.model_provider == "openai" else self.ollama_model
