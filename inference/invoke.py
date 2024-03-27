import boto3
import json
import ast
from uuid import uuid4

from config import global_config

payload = json.dumps({"s3_image_key": "Audi_S6.jpg"})

response = boto3.client("sagemaker-runtime").invoke_endpoint(
    EndpointName=global_config.deploy.endpoint_name,
    CustomAttributes=str(uuid4()),
    ContentType="application/json",
    Accept="application/json",
    Body=payload,
)

ast.literal_eval(response['Body'].read().decode("utf-8"))["Predicted"]
# Returns "Audi S6 Sedan 2011"
# Can test via SageMaker console too