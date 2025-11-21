from __future__ import annotations

import importlib
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import torch 
import random 
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

def locate(path: str):
    """Import and return an object given a dotted path."""
    if "." not in path:
        raise ValueError(f"Expected dotted path for import, got '{path}'")
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def to_container(cfg_section: DictConfig | Dict | None) -> Dict[str, Any]:
    if cfg_section is None:
        return {}
    if isinstance(cfg_section, DictConfig):
        return OmegaConf.to_container(cfg_section, resolve=True)  # type: ignore[arg-type]
    return cfg_section


def instantiate(value: Any) -> Any:
    """Recursively instantiate dicts that declare a `_target_` key."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        if "_target_" in value:
            target = locate(value["_target_"])
            kwargs = {k: instantiate(v) for k, v in value.items() if k != "_target_"}
            return target(**kwargs)
        return {k: instantiate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [instantiate(v) for v in value]
    return value


def parse_args(args_cfg: DictConfig | Dict | None) -> Dict[str, Any]:
    """Convert an OmegaConf section into a plain dict with instantiated targets."""
    args = to_container(args_cfg)
    return {k: instantiate(v) for k, v in args.items()}


def set_seed(seed: int) -> None:    
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_or_path: DictConfig | str):
    if isinstance(cfg_or_path, DictConfig):
        return cfg_or_path
    base_config = OmegaConf.load("configs/base_config.yaml")
    cfg_path = Path(cfg_or_path)
    return OmegaConf.merge(base_config, OmegaConf.load(cfg_path))


def load_full_dataset(
    dataset_cfg: DictConfig, additional_config: Dict[str, Any] = {}
) -> Tuple[Any, Dict[str, Any]]:
    if "class_path" not in dataset_cfg:
        raise KeyError("Dataset config requires 'class_path'.")
    class_path = dataset_cfg.class_path
    dataset_cls = locate(class_path)
    args_cfg = dataset_cfg.get("args", {})
    args = parse_args(args_cfg)
    for k, v in additional_config.items():
        args[k] = v
    return dataset_cls, args


def build_model(cfg: DictConfig):
    model_cfg = cfg.get("model")
    if model_cfg:
        loader_path = getattr(model_cfg, "class_path", None)
        if loader_path is None:
            raise KeyError("Model config requires 'class_path'.")
        loader_fn = locate(loader_path)
        loader_args = parse_args(model_cfg.get("args", {}))
        return loader_fn(**loader_args)

    raise ValueError("No model configuration found in cfg.")