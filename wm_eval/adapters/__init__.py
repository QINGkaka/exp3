from .fastwam import FastWAMAdapter
from .openwam import OpenWAMAdapter

ADAPTERS = {
    "fastwam": FastWAMAdapter,
    "openwam": OpenWAMAdapter,
}

__all__ = ["ADAPTERS", "FastWAMAdapter", "OpenWAMAdapter"]

