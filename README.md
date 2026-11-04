# LLM4ACOE
LLM-based framework that automates the ontology engineering process with LLM-powered RAG-enhanced agent-based role-playing simulation.

## Architecture 
![LLM4ACOE](https://github.com/user-attachments/assets/1a43d2f5-c5c3-421b-ad1a-61af613d3fc2)

## Quick Install
Python 3.12
For Windows
```bash
py -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
py -m pip install -r requirements.txt
```

For Unix/macOS
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
python3 -m pip install -r requirements.txt
```

## Model and provider configuration
Copy the sample environment file and set your own values:

```bash
cp .env.example .env
```

Then edit `.env` for global credentials and endpoint configuration only:

```dotenv
OPENAI_API_KEY=your_openai_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_VERSION=2024-02-01

AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_BASE_URL=https://api.anthropic.com

GOOGLE_API_KEY=your_google_key
GOOGLE_API_BASE_URL=

NVIDIA_API_KEY=your_nvidia_key
NVIDIA_API_BASE_URL=https://integrate.api.nvidia.com/v1
```

Choose the run-specific provider, model, and any model kwargs at runtime via argparse, for example:

```bash
python LLM4ACOE.py \
  --provider azure-openai \
  --model gpt-4o-mini \
  --temperature 0.2 \
  --max-tokens 512 \
  --embedding-provider huggingface \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Use `--model-kwargs '{"top_p": 0.9, "seed": 42}'` to pass extra runtime options as a JSON object.

Supported LLM providers are `openai`, `azure-openai`, `anthropic`, `google`, and `nvidia`.
Supported embedding providers are `huggingface`, `openai`, and `azure-openai`.
