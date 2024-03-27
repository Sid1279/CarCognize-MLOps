from abc import ABC # Python OOP ;)
from omegaconf import OmegaConf
from config import global_config

class PipelineStage(ABC):
    def __init__(self, stage_config, stage_dir):
        self.config = OmegaConf.merge(global_config.common, stage_config)
        print(self.config.name, f"stage config: {self.config}")

        self.stage_dir = stage_dir
        self.sagemaker_root = "/opt/ml/processing"

        self.inputs = None
        self.outputs = None
        self.processor = None
        self.estimator = None
        
    def set_inputs(self):
        pass

    def set_outputs(self):
        pass

    def run(self):
        pass

    def set_processor(self):
        pass
    
    def set_estimator(self):
        pass
