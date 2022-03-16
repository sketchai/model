from typing import Dict
import collections.abc
import yaml


def yaml_to_dict(file_path: str) -> Dict:
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data
    return None
