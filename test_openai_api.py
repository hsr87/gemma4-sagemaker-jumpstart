"""Test a Gemma 4 SageMaker endpoint via the OpenAI-compatible API.

Reference:
- https://aws.amazon.com/blogs/machine-learning/announcing-openai-compatible-api-support-for-amazon-sagemaker-ai-endpoints/

SageMaker exposes an OpenAI-compatible surface at:
    https://runtime.sagemaker.<REGION>.amazonaws.com/endpoints/<ENDPOINT>/openai/v1
Auth is a short-lived bearer token minted from your AWS credentials with
`sagemaker.core.token_generator.generate_token`, so the OpenAI SDK can talk to
the endpoint directly over HTTPS (no boto3 SigV4 needed at call time).
"""
import json
import os
import sys
from datetime import timedelta

import httpx
from openai import OpenAI
from sagemaker.core.token_generator import generate_token

import config


def resolve_endpoint_name() -> str:
    """Prefer $GEMMA4_ENDPOINT_NAME, then endpoint_info.json, then config default."""
    env = os.environ.get("GEMMA4_ENDPOINT_NAME")
    if env:
        return env
    if os.path.exists(config.ENDPOINT_INFO_FILE):
        with open(config.ENDPOINT_INFO_FILE) as f:
            return json.load(f)["endpoint_name"]
    return config.ENDPOINT_NAME


class SageMakerBearerAuth(httpx.Auth):
    """Re-mint the bearer token on every request so long runs never expire."""

    def __init__(self, region: str):
        self.region = region

    def auth_flow(self, request):
        token = generate_token(region=self.region, expiry=timedelta(minutes=5))
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def make_client(endpoint_name: str) -> OpenAI:
    base_url = (
        f"https://runtime.sagemaker.{config.REGION}.amazonaws.com"
        f"/endpoints/{endpoint_name}/openai/v1"
    )
    print(f"base_url: {base_url}")
    return OpenAI(
        base_url=base_url,
        # Initial key; auth_flow below refreshes it per request.
        api_key=generate_token(region=config.REGION, expiry=timedelta(minutes=5)),
        http_client=httpx.Client(auth=SageMakerBearerAuth(config.REGION), timeout=120.0),
    )


def test_chat_completion(client: OpenAI, model: str) -> None:
    print("\n--- Test 1: non-streaming chat.completions ---")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "In one sentence, what is Amazon SageMaker JumpStart?"},
        ],
        max_tokens=128,
        temperature=0.2,
    )
    print("response:", resp.choices[0].message.content)
    if resp.usage:
        print(f"usage: prompt={resp.usage.prompt_tokens} "
              f"completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")


def test_streaming(client: OpenAI, model: str) -> None:
    print("\n--- Test 2: streaming chat.completions ---")
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Count from 1 to 5, separated by commas."},
        ],
        max_tokens=64,
        stream=True,
    )
    print("streamed: ", end="", flush=True)
    chunks = 0
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            chunks += 1
    print(f"\n({chunks} content chunks received)")


def main() -> None:
    endpoint_name = resolve_endpoint_name()
    print(f"=== Testing endpoint '{endpoint_name}' via OpenAI-compatible API ===")
    client = make_client(endpoint_name)

    # The model field is routed through to the container. The JumpStart vLLM
    # container serves the model under "/opt/ml/model"; per the blog you can also
    # leave it empty ("") and the endpoint resolves to its single served model.
    model = os.environ.get("GEMMA4_SERVED_MODEL", "")
    try:
        test_chat_completion(client, model)
        test_streaming(client, model)
    except Exception as exc:  # surface the real HTTP error body if present
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        body = getattr(getattr(exc, "response", None), "text", None)
        if body:
            print(f"response body: {body}", file=sys.stderr)
        sys.exit(1)

    print("\n=== All OpenAI-compatible API tests passed ===")


if __name__ == "__main__":
    main()
