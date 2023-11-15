from logging import Logger
from typing import Any, Dict, Optional

from dataclasses import fields, is_dataclass


def update_config_with_dict(config: Any, params: Dict, log: Optional[Logger] = None) -> None:
    # ToDo: Fix error where "nested" params are sometimes saved wrong in string form instead of dict
    if isinstance(params, str):
        params = eval(params)
    assert is_dataclass(config)
    for field in fields(config):
        if field.name not in params:
            continue
        val = getattr(config, field.name)
        if is_dataclass(val):
            update_config_with_dict(val, params[field.name], log)
        elif val != params[field.name]:
            if val is not None:
                setattr(config, field.name, type(val)(params[field.name]))
            else:
                setattr(config, field.name, params[field.name])
            if log:
                log.info(f"Setting {field.name} to {params[field.name]}")


def flatten(config: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    result = {}
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            result.update(flatten(v, parent_key=new_key, sep=sep))
        else:
            result[new_key] = v
    return result


def unflatten(config: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    result = dict()

    for k, v in config.items():
        parts = k.split(sep)
        sub_dict = result
        for part in parts[:-1]:
            if part not in sub_dict:
                sub_dict[part] = dict()
            sub_dict = sub_dict[part]
        sub_dict[parts[-1]] = v

    return result
