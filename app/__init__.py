from config import global_config

stages = []
for i in global_config.keys():
    stages.append(i)
print("Pipeline Stages: ", stages)