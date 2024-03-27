from pathlib import Path
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent / 'config'

globals()["global_config"] = OmegaConf.load(CONFIG_DIR.glob("*.yaml").__next__())

# bringing all vars declared in the config file to global scope
# print(global_config)