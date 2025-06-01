import json
import os
import yaml


def load_config(path: str) -> dict:
    """
    Load a configuration file (YAML or JSON) from `path` and return a Python dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    _, ext = os.path.splitext(path.lower())

    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise ValueError("PyYAML is required to load YAML config files.")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported config extension. Use .yaml, .yml, or .json.")

    if not isinstance(data, dict):
        raise ValueError("Config file root must be a dictionary.")
    return data