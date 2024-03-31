import tarfile
from sagemaker.pytorch import PyTorchModel

from config import global_config as cfg
train_cfg = cfg.train.stage
deploy_cfg = cfg.deploy.stage

model_key = f"/opt/ml/processing/deploying_input/{train_cfg.model_name}.pth"
tar_archive = f"/opt/ml/processing/deploying_input/{train_cfg.model_name}.tar.gz"

with tarfile.open(tar_archive, 'w:gz') as tar:
    tar.add(model_key, arcname=f"{train_cfg.model_name}.pth")

pytorch_model = PyTorchModel(
    model_data=tar_archive,
    role=cfg.common.role,
    framework_version=deploy_cfg.framework_version,
    entry_point=deploy_cfg.entry_script,
    py_version=deploy_cfg.py_version
)

predictor = pytorch_model.deploy(
    endpoint_name=deploy_cfg.endpoint_name,
    initial_instance_count=deploy_cfg.initial_instance_count,
    instance_type=deploy_cfg.instance_type
)
