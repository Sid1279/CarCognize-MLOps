from pipeline_stage import PipelineStage
import app

import os

from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.pytorch import PyTorch

class ProcessingStage(PipelineStage):
    def set_inputs(self):
        self.inputs = [
            ProcessingInput(
                source=self.config.data_directory.dataset,
                destination=f"{self.sagemaker_root}/{self.config.pipeline_name}/input/",
                input_name=self.config.processing_input.input_name,
                s3_data_type=self.config.s3_data_type,
                s3_input_mode=self.config.s3_input_mode,
                s3_data_distribution_type=self.config.s3_data_distribution_type,
            )
        ]
    
    def set_outputs(self):
        self.outputs = [
            ProcessingOutput(
                source=f"{self.sagemaker_root}/{self.config.pipeline_name}/output/",
                destination=self.config.data_directory.preprocessed,
                output_name=self.config.processing_output.output_name,
                s3_upload_mode=self.config.s3_upload_mode,
            )
        ]
    
    def set_processor(self):
        self.processor = FrameworkProcessor(
            estimator_cls=PyTorch,
            framework_version=self.config.processor.framework_version,
            py_version=self.config.processor.py_version,
            role=self.config.role,
            instance_count=self.config.processor.instance_count,
            instance_type=self.config.processor.instance_type,
            max_runtime_in_seconds=self.config.processor.max_runtime_in_seconds,
        )

    def create_stage(self):
        self.set_inputs()
        self.set_outputs()
        self.set_processor()

        processing_step = ProcessingStep(
            name=self.config.step_name,
            code=self.config.code,
            source_dir=str(self.stage_dir),
            inputs=self.inputs,
            outputs=self.outputs,
            processor=self.processor,
        )

        return processing_step

class TrainingStage(PipelineStage):
    def set_inputs(self):
        self.inputs = {
            self.config.data_directory.preprocessed: TrainingInput(
                s3_data=app.processing.properties.ProcessingOutputConfig.Outputs[
                    self.config.data_directory.preprocessed
                ].S3Output.S3Uri,
                content_type='application/json',
            ),
        }
    
    def set_estimator(self):
        hyperparameters = {
            "model_type": self.config.model_type,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "loss": self.config.loss,
            "metrics": self.config.metrics,
            "optimizer_type": self.config.optimizer_type,
            "optimizer_lr": self.config.optimizer_lr,
        }

        self.estimator = PyTorch(
            entry_point=os.path.join(self.stage_dir, "stage.py"),
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            role=self.config.role,
            script_mode=True,
            hyperparameters=hyperparameters
        )
    
    def create_stage(self):
        self.set_inputs()
        self.set_estimator()

        training_step = TrainingStep(
            name=self.config.step_name,
            estimator=self.estimator,
            inputs=self.inputs
        )
        return training_step

class DeploymentStage(PipelineStage):
    def set_inputs(self):
        self.inputs = [
            ProcessingInput(
                source=app.training.properties.ModelArtifacts.S3ModelArtifacts,
                destination=f"{self.sagemaker_root}/{self.config.pipeline_name}/model/",
                input_name="deploy_model_input",
                s3_data_type=self.config.s3_data_type,
                s3_input_mode=self.config.s3_input_mode,
                s3_data_distribution_type=self.config.s3_data_distribution_type,
            ),
        ]

    def set_processor(self):
        self.processor = FrameworkProcessor(
            estimator_cls=PyTorch,
            framework_version=self.config.processor.framework_version,
            py_version=self.config.processor.py_version,
            role=self.config.role,
            instance_count=self.config.processor.instance_count,
            instance_type=self.config.processor.instance_type,
            max_runtime_in_seconds=self.config.processor.max_runtime_in_seconds,
        )

    def create_stage(self):
        self.set_inputs()
        self.set_outputs()
        self.set_processor()

        deployment_step = ProcessingStep(
            name=self.config.step_name,
            code=self.config.code,
            source_dir=str(self.stage_dir),
            inputs=self.inputs,
            outputs=self.outputs,
            processor=self.processor,
        )

        return deployment_step