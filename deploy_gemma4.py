"""Deploy a Gemma 4 model from SageMaker JumpStart to a real-time endpoint.

Uses the SageMaker Python SDK v3 `ModelBuilder` workflow:
    ModelBuilder(model=<jumpstart-id>).build().deploy()

Reference:
- https://aws.amazon.com/about-aws/whats-new/2026/04/gemma-4-models-on-sagemaker-jumpstart/

The endpoint is created with the OpenAI-compatible API enabled (JumpStart LLM
containers expose `/openai/v1/...` automatically), so test_openai_api.py can
talk to it with the OpenAI SDK afterwards.
"""
import json
import time

import boto3
from sagemaker.serve.model_builder import ModelBuilder

import config


def write_info(endpoint_name: str) -> dict:
    info = {
        "endpoint_name": endpoint_name,
        "model_id": config.MODEL_ID,
        "instance_type": config.INSTANCE_TYPE,
        "region": config.REGION,
    }
    with open(config.ENDPOINT_INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
    return info


def wait_in_service(endpoint_name: str, timeout_s: int = 1800) -> str:
    """Poll endpoint status, tolerating transient API/network errors."""
    sm = boto3.client("sagemaker", region_name=config.REGION)
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        except Exception as exc:  # transient connection blips, throttling, etc.
            print(f"  (poll retry after transient error: {type(exc).__name__})")
            time.sleep(15)
            continue
        if status != last:
            print(f"  status: {status}")
            last = status
        if status == "InService":
            return status
        if status in ("Failed", "OutOfService", "DeleteFailed"):
            raise RuntimeError(f"Endpoint entered terminal status: {status}")
        time.sleep(20)
    raise TimeoutError(f"Endpoint not InService within {timeout_s}s (last={last})")


def main() -> None:
    if not config.ROLE_ARN:
        raise SystemExit(
            "SAGEMAKER_ROLE_ARN is not set. Provide your SageMaker execution role, e.g.:\n"
            '  export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/service-role/<your-sagemaker-role>"'
        )

    print("=== Deploying Gemma 4 to SageMaker JumpStart ===")
    print(f"  model_id      : {config.MODEL_ID} ({config.MODEL_VERSION})")
    print(f"  instance_type : {config.INSTANCE_TYPE} x {config.INSTANCE_COUNT}")
    print(f"  region        : {config.REGION}")
    print(f"  endpoint_name : {config.ENDPOINT_NAME}")
    print(f"  role_arn      : {config.ROLE_ARN}")
    print()

    builder = ModelBuilder(
        model=config.MODEL_ID,
        role_arn=config.ROLE_ARN,
        instance_type=config.INSTANCE_TYPE,
    )
    # Gemma is a gated model -> must accept the EULA to deploy.
    builder.accept_eula = True

    print("Building model (resolving JumpStart artifacts + container)...")
    builder.build()

    print("Deploying endpoint (this typically takes 10-15 minutes)...")
    start = time.time()
    # Kick off creation without blocking on the SDK's internal waiter, which can
    # crash on a single transient network error. We wait with our own tolerant loop.
    endpoint = builder.deploy(
        endpoint_name=config.ENDPOINT_NAME,
        initial_instance_count=config.INSTANCE_COUNT,
        instance_type=config.INSTANCE_TYPE,
        accept_eula=True,
        wait=False,
    )

    # Resolve the actual endpoint name from whatever the SDK returned, and record
    # it immediately so the test/cleanup scripts work even if waiting is interrupted.
    endpoint_name = getattr(endpoint, "endpoint_name", None) or config.ENDPOINT_NAME
    info = write_info(endpoint_name)

    wait_in_service(endpoint_name)
    elapsed = time.time() - start

    print()
    print(f"=== Endpoint InService in {elapsed/60:.1f} min ===")
    print(json.dumps(info, indent=2))
    print(f"\nWrote endpoint info -> {config.ENDPOINT_INFO_FILE}")
    print("Next: python test_openai_api.py")


if __name__ == "__main__":
    main()
