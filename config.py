import json

with open("config.json", "r") as file:
    config = json.load(file)

dataset_name = config["dataset_name"]
model_type = config["model_type"]
metric_type = config["metric_type"]
pipeline_type = config["pipeline_type"]
pipeline_order = config["pipeline_order"]

config["log_file_path"] = config["paths"]["log_file"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["train_file"] = config["paths"]["train_data"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["test_file"] = config["paths"]["test_data"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)

config["metric_file"] = config["paths"]["metric_output"].format(
    dataset_name=dataset_name,
    model_type=model_type,
    metric_type=metric_type
)


