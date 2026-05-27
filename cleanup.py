"""Delete the Gemma 4 endpoint, endpoint-config, and model to stop billing.

Run this when you are done testing:
    python cleanup.py
"""
import json
import os

import boto3

import config


def resolve_endpoint_name() -> str:
    env = os.environ.get("GEMMA4_ENDPOINT_NAME")
    if env:
        return env
    if os.path.exists(config.ENDPOINT_INFO_FILE):
        with open(config.ENDPOINT_INFO_FILE) as f:
            return json.load(f)["endpoint_name"]
    return config.ENDPOINT_NAME


def main() -> None:
    sm = boto3.client("sagemaker", region_name=config.REGION)
    name = resolve_endpoint_name()
    print(f"Cleaning up endpoint '{name}' in {config.REGION}...")

    # Capture model names referenced by the endpoint config before deleting.
    model_names = []
    try:
        ep = sm.describe_endpoint(EndpointName=name)
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        model_names = [v["ModelName"] for v in cfg.get("ProductionVariants", [])]
    except sm.exceptions.ClientError as e:
        print(f"  (could not describe endpoint: {e})")

    for action, fn in [
        ("endpoint", lambda: sm.delete_endpoint(EndpointName=name)),
        ("endpoint-config", lambda: sm.delete_endpoint_config(EndpointConfigName=name)),
    ]:
        try:
            fn()
            print(f"  deleted {action}: {name}")
        except sm.exceptions.ClientError as e:
            print(f"  skip {action}: {e}")

    for model_name in model_names:
        try:
            sm.delete_model(ModelName=model_name)
            print(f"  deleted model: {model_name}")
        except sm.exceptions.ClientError as e:
            print(f"  skip model {model_name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
