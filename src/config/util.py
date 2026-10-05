from typing import Any


def flatten(config: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    result = {}
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            result.update(flatten(v, parent_key=new_key, sep=sep))
        else:
            result[new_key] = v
    return result
