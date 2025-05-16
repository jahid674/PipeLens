# config.py
import json

with open("config.json", "r") as file:
    config = json.load(file)

# Create formatted versions of templated strings
dataset_name = config["dataset_name"]
model_type = config["model_type"]
metric_type = config["metric_type"]
algortihm_type = config["algorithm_type"]


config["log_file_path"] = config["log_file_path"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["train_file"] = config["train_file_template"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["test_file"] = config["test_file_template"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["metric_file"] = config["metric_file_template"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

# Optional: expose key configs as globals
pipeline_order = config["pipeline_order"]
pipeline_type = config["pipeline_type"]
