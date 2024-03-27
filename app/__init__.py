from config import global_config
from pathlib import Path

APP_DIR = Path(__file__).parent

stages = []
for i in global_config.keys():
    if i != "common":
        stages.append(i)
print("Pipeline Stages: ", stages)