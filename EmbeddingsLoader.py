import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


class EmbeddingsLoader:
    @staticmethod
    def _get_runtime_value(*env_names):
        for env_name in env_names:
            value = os.getenv(env_name)
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def load_embeddings(provider=None, model=None, fallback_model=None):
        provider_name = (provider or "huggingface").strip().lower()
        primary_model = model or "sentence-transformers/all-MiniLM-L6-v2"
        secondary_model = fallback_model or primary_model

        if provider_name in {"huggingface", "sentence-transformers", "hf"}:
            try:
                return HuggingFaceEmbeddings(model_name=primary_model)
            except Exception as exc:
                if secondary_model and secondary_model != primary_model:
                    try:
                        return HuggingFaceEmbeddings(model_name=secondary_model)
                    except Exception:
                        pass
                raise RuntimeError(
                    "Unable to initialize Hugging Face embeddings with the configured model(s): "
                    f"{primary_model}, {secondary_model}."
                ) from exc

        if provider_name in {"openai", "azure-openai", "azure"}:
            azure_mode = provider_name in {"azure-openai", "azure"}
            if azure_mode:
                api_key = EmbeddingsLoader._get_runtime_value("AZURE_OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required when using Azure OpenAI embeddings.")
                azure_endpoint = EmbeddingsLoader._get_runtime_value("AZURE_OPENAI_ENDPOINT")
                if azure_endpoint is None:
                    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required when using Azure OpenAI embeddings.")
                api_version = EmbeddingsLoader._get_runtime_value("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION") or "2024-02-01"
                selected_model = model or "text-embedding-3-small"
                return AzureOpenAIEmbeddings(
                    model=selected_model,
                    azure_endpoint=azure_endpoint,
                    openai_api_version=api_version,
                    openai_api_type="azure",
                    api_key=api_key,
                )

            api_key = EmbeddingsLoader._get_runtime_value("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI embeddings.")
            selected_model = model or "text-embedding-3-small"
            kwargs = {
                "model": selected_model,
                "openai_api_key": api_key,
            }
            api_base = EmbeddingsLoader._get_runtime_value("OPENAI_API_BASE")
            if api_base:
                kwargs["openai_api_base"] = api_base
            api_version = EmbeddingsLoader._get_runtime_value("OPENAI_API_VERSION")
            if api_version:
                kwargs["openai_api_version"] = api_version
            return OpenAIEmbeddings(**kwargs)

        raise ValueError(
            "Unsupported embedding provider: "
            f"{provider_name}. Supported providers are huggingface, openai, and azure-openai."
        )
