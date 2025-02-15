# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg
from .lazy import LazyCall, LazyConfig

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "LazyCall",
    "LazyConfig",
]
