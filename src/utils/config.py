from __future__ import annotations

from pathlib import Path

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as error:
            raise AttributeError(key) from error
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[key] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_config_node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    visualization = raw.get("visualization", {})
    if "num_fixed_samples" not in visualization and "num_fixed_test_samples" in visualization:
        visualization["num_fixed_samples"] = visualization["num_fixed_test_samples"]
    return _to_config_node(raw)


def resolve_path(project_root: str | Path, maybe_relative_path: str | Path) -> Path:
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path
