"""Shared configuration for the Gemma 4 SageMaker JumpStart demo.

Values can be overridden with environment variables so the deploy / test /
cleanup scripts all agree on the same endpoint.
"""
import os

# Region the endpoint lives in.
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Gemma 4 JumpStart model IDs (from `list_jumpstart_models()`):
#   huggingface-vlm-gemma-4-e4b-it      -> ml.g6e.2xlarge  (smallest / cheapest)
#   huggingface-vlm-gemma-4-26b-a4b-it  -> ml.g6.12xlarge
#   huggingface-vlm-gemma-4-31b-it      -> ml.g6.24xlarge
MODEL_ID = os.environ.get("GEMMA4_MODEL_ID", "huggingface-vlm-gemma-4-e4b-it")
MODEL_VERSION = os.environ.get("GEMMA4_MODEL_VERSION", "*")
INSTANCE_TYPE = os.environ.get("GEMMA4_INSTANCE_TYPE", "ml.g6e.2xlarge")
INSTANCE_COUNT = int(os.environ.get("GEMMA4_INSTANCE_COUNT", "1"))

# SageMaker execution role used to create the model / endpoint.
# Set via the SAGEMAKER_ROLE_ARN env var, e.g.:
#   export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/service-role/<your-sagemaker-role>"
# Left empty by default so no account-specific ARN is committed; deploy validates it.
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")

# Endpoint name. Deploy writes the resolved name to ENDPOINT_INFO_FILE; the test
# and cleanup scripts read it back from there (or from $GEMMA4_ENDPOINT_NAME).
ENDPOINT_NAME = os.environ.get("GEMMA4_ENDPOINT_NAME", "gemma4-e4b-openai-demo")

# Where deploy records the live endpoint name for the other scripts.
ENDPOINT_INFO_FILE = os.path.join(os.path.dirname(__file__), "endpoint_info.json")
