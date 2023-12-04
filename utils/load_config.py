
## load_config.py

import os

import yaml


def load_config(model_name):
    config_file_path = os.path.join("models", model_name, "configs", f"config_{model_name.lower()}.yaml")

    config = {}
    with open(config_file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config

def substitute_variables(config):
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
        elif isinstance(value, dict):
            config[key] = substitute_variables(value)

    return config