import json
import os
import yaml


def load_config(path: str) -> dict:
    """
    Load a configuration file (YAML or JSON) from the given path and return it as a Python dictionary.

    This utility supports `.yaml`, `.yml`, and `.json` file formats. It performs file existence checks,
    handles file extension parsing, and ensures the root object of the loaded config is a dictionary.

    Parameters
    ----------
    path : str
        The path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration data as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.

    ValueError
        - If the file extension is unsupported.
        - If the file content is not a dictionary.
        - If PyYAML is not installed when loading a YAML file.
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
