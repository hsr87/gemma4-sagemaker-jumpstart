# Gemma 4 on SageMaker JumpStart + OpenAI-compatible API

> **개요 (한국어)**
> 이 저장소는 **Amazon SageMaker JumpStart로 Gemma 4 모델을 손쉽게 배포**하고,
> 배포된 엔드포인트가 **OpenAI 호환 API 포맷으로 정상 동작하는지 테스트**하기 위한 예제입니다.
> `deploy_gemma4.py`로 배포하고, `test_openai_api.py`에서 OpenAI Python SDK로 호출(채팅·스트리밍)해
> 확인한 뒤, `cleanup.py`로 리소스를 정리하는 end-to-end 흐름을 제공합니다.
> 자세한 단계별 가이드는 [`gemma4_guide.md`](./gemma4_guide.md)를 참고하세요.

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
