# llm_builder.py
from enum import Enum
from typing import Optional, Dict, Any
import os

from dotenv import load_dotenv
from langchain.llms import OpenAI, AzureOpenAI, Anthropic, GoogleGemini
# from your_grok_module import Grok  # hypothetical

# Load .env into os.environ
load_dotenv()


class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure_openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    GROK = "grok"


class LLMBuilder:
    def __init__(self):
        self._provider: Optional[Provider] = None
        self._config: Dict[str, Any] = {}

    def with_openai(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
    ) -> "LLMBuilder":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment")
        self._provider = Provider.OPENAI
        self._config = {
            "openai_api_key": api_key,
            "model_name":      model_name,
            "temperature":     temperature,
        }
        return self

    def with_azure_openai(
        self,
        deployment_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
    ) -> "LLMBuilder":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_API_BASE")
        deploy = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not (api_key and api_base and deploy):
            raise ValueError(
                "Missing one of AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, "
                "or AZURE_OPENAI_DEPLOYMENT_NAME in environment"
            )
        self._provider = Provider.AZURE
        self._config = {
            "openai_api_key": api_key,
            "openai_api_base": api_base,
            "deployment_name": deploy,
            "model_name":      model_name or deploy,
            "temperature":     temperature,
        }
        return self

    def with_claude(
        self,
        model: str = "claude-2",
        temperature: float = 0.7,
    ) -> "LLMBuilder":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in environment")
        self._provider = Provider.CLAUDE
        self._config = {
            "anthropic_api_key": api_key,
            "model":             model,
            "temperature":       temperature,
        }
        return self

    def with_gemini(
        self,
        model: str = "gemini-pro",
        temperature: float = 0.7,
    ) -> "LLMBuilder":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")
        self._provider = Provider.GEMINI
        self._config = {
            "google_api_key": api_key,
            "model":           model,
            "temperature":     temperature,
        }
        return self

    def with_grok(
        self,
        model: str = "grok-1",
        temperature: float = 0.7,
    ) -> "LLMBuilder":
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("Missing GROK_API_KEY in environment")
        self._provider = Provider.GROK
        self._config = {
            "grok_api_key": api_key,
            "model":         model,
            "temperature":   temperature,
        }
        return self

    def build(self):
        if self._provider == Provider.OPENAI:
            return OpenAI(**self._config)
        elif self._provider == Provider.AZURE:
            return AzureOpenAI(**self._config)
        elif self._provider == Provider.CLAUDE:
            return Anthropic(**self._config)
        elif self._provider == Provider.GEMINI:
            return GoogleGemini(**self._config)
        elif self._provider == Provider.GROK:
            # return Grok(**self._config)
            raise NotImplementedError("Grok support stubbed out")
        else:
            raise ValueError(
                "No LLM configured; call a `.with_*()` method first.")
