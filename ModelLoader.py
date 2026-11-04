import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class ModelLoader:
    @staticmethod
    def _get_runtime_value(*env_names):
        for env_name in env_names:
            value = os.getenv(env_name)
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def load_llm(provider=None, model=None, temperature=0.5, model_kwargs=None):
        provider_name = (provider or "openai").strip().lower()
        model_kwargs = dict(model_kwargs or {})

        if provider_name in {"openai", "azure-openai", "azure"}:
            default_model = "gpt-4o"
            selected_model = model or default_model
            api_key = ModelLoader._get_runtime_value("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
            api_base = ModelLoader._get_runtime_value("AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE")
            api_version = ModelLoader._get_runtime_value("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION")

            if provider_name in {"azure-openai", "azure"}:
                if api_key is None:
                    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required when using Azure OpenAI.")
                if api_base is None:
                    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required when using Azure OpenAI.")
                kwargs = {
                    "model_name": selected_model,
                    "deployment_name": selected_model,
                    "azure_endpoint": api_base,
                    "api_key": api_key,
                    "openai_api_version": api_version or "2024-02-01",
                    "openai_api_type": "azure",
                    "temperature": temperature,
                }
                kwargs.update(model_kwargs)
                return AzureChatOpenAI(**kwargs)

            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI.")
            kwargs = {
                "model": selected_model,
                "openai_api_key": api_key,
                "temperature": temperature,
            }
            if api_base:
                kwargs["openai_api_base"] = api_base
            if api_version:
                kwargs["openai_api_version"] = api_version
            kwargs.update(model_kwargs)
            return ChatOpenAI(**kwargs)

        if provider_name in {"anthropic", "claude"}:
            default_model = "claude-3-5-sonnet-20241022"
            selected_model = model or default_model
            api_key = ModelLoader._get_runtime_value("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required when using Anthropic.")

            kwargs = {
                "model": selected_model,
                "temperature": temperature,
                "anthropic_api_key": api_key,
            }
            api_base = ModelLoader._get_runtime_value("ANTHROPIC_BASE_URL")
            if api_base:
                kwargs["base_url"] = api_base
            kwargs.update(model_kwargs)
            return ChatAnthropic(**kwargs)

        if provider_name in {"google", "gemini"}:
            default_model = "gemini-1.5-pro"
            selected_model = model or default_model
            api_key = ModelLoader._get_runtime_value("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("GOOGLE_API_KEY environment variable is required when using Google Gemini.")

            kwargs = {
                "model": selected_model,
                "temperature": temperature,
                "google_api_key": api_key,
            }
            api_base = ModelLoader._get_runtime_value("GOOGLE_API_BASE_URL")
            if api_base:
                kwargs["base_url"] = api_base
            kwargs.update(model_kwargs)
            return ChatGoogleGenerativeAI(**kwargs)

        if provider_name == "nvidia":
            default_model = "mistralai/mixtral-8x22b-instruct-v0.1"
            selected_model = model or default_model
            api_key = ModelLoader._get_runtime_value("NVIDIA_API_KEY")
            if api_key is None:
                raise ValueError("NVIDIA_API_KEY environment variable is required when using NVIDIA.")

            kwargs = {
                "model": selected_model,
                "temperature": temperature,
                "api_key": api_key,
            }
            api_base = ModelLoader._get_runtime_value("NVIDIA_API_BASE_URL")
            if api_base:
                kwargs["base_url"] = api_base
            kwargs.update(model_kwargs)
            return ChatNVIDIA(**kwargs)

        raise ValueError(
            "Unsupported LLM provider: "
            f"{provider_name}. Supported providers are openai, azure-openai, anthropic, google, and nvidia."
        )

        