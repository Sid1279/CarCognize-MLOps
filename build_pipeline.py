from pathlib import Path
from sagemaker.workflow.pipeline import Pipeline

from config import global_config
from pipeline_classes import ProcessingStage, TrainingStage, DeploymentStage

processing = ProcessingStage(global_config.preprocess, Path.cwd() / "a_preprocess")
training = TrainingStage(global_config.train, Path.cwd() / "b_train")
deployment = DeploymentStage(global_config.deploy, Path.cwd() / "c_deploy")

pipeline_steps = [
    processing.create_stage(),
    training.create_stage(),
    deployment.create_stage()
]


pipeline = Pipeline(
    name=global_config.common.pipeline_name,
    steps=pipeline_steps,
)

pipeline.upsert(role=global_config.common.role)