# Gemma 4 on SageMaker JumpStart + OpenAI-compatible API

Deploys a **Gemma 4** model from Amazon SageMaker JumpStart to a real-time
endpoint and tests it through SageMaker's **OpenAI-compatible API**.

References:
- [Gemma 4 models on SageMaker JumpStart](https://aws.amazon.com/about-aws/whats-new/2026/04/gemma-4-models-on-sagemaker-jumpstart/)
- [OpenAI-compatible API support for SageMaker AI endpoints](https://aws.amazon.com/blogs/machine-learning/announcing-openai-compatible-api-support-for-amazon-sagemaker-ai-endpoints/)

## Available Gemma 4 variants

| JumpStart model ID | Default instance | Notes |
|---|---|---|
| `huggingface-vlm-gemma-4-e4b-it` | `ml.g6e.2xlarge` | smallest / cheapest (used here) |
| `huggingface-vlm-gemma-4-26b-a4b-it` | `ml.g6.12xlarge` | mid-size MoE |
| `huggingface-vlm-gemma-4-31b-it` | `ml.g6.24xlarge` | largest |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires AWS credentials with SageMaker permissions (deploy + `sagemaker:InvokeEndpoint`
+ `sagemaker:CallWithBearerToken`). Edit `config.py` or set env vars to change the
model, instance type, region, or execution role.

## Usage

```bash
python deploy_gemma4.py     # deploy endpoint (~10-15 min), writes endpoint_info.json
python test_openai_api.py   # call it via the OpenAI SDK (chat + streaming)
python cleanup.py           # delete endpoint/config/model to stop billing
```

## How it works

- **Deploy** uses the SageMaker Python SDK v3 `ModelBuilder` workflow:
  `ModelBuilder(model="huggingface-vlm-gemma-4-e4b-it").build().deploy()`.
  `accept_eula=True` is set because Gemma is a gated model.
- **Test** points the OpenAI Python client at
  `https://runtime.sagemaker.<region>.amazonaws.com/endpoints/<endpoint>/openai/v1`
  and authenticates with a short-lived bearer token from
  `sagemaker.core.token_generator.generate_token`. An `httpx.Auth` re-mints the
  token per request so long sessions don't expire.

## Cost

The endpoint bills per hour while `InService` (E4B on `ml.g6e.2xlarge` ≈ $2.2/hr,
on-demand, us-east-1). Run `python cleanup.py` when done.
