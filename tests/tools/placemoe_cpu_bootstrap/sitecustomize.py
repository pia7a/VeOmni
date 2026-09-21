"""CPU review harness: bypass unrelated eager accelerator package registration."""

import importlib.machinery
import os
import sys
import types
from pathlib import Path


root = Path(os.environ["PLACEMOE_SOURCE"])
sys.path.insert(0, str(root))
for name in ("veomni", "veomni.distributed", "veomni.distributed.moe"):
    module = types.ModuleType(name)
    module.__path__ = [str(root.joinpath(*name.split(".")))]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module
