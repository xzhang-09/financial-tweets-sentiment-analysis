import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional for baseline-only environments
    torch = None


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str | Path) -> Dict:
    with open(path) as file_pointer:
        return json.load(file_pointer)


def save_dict(d: Dict, path: str | Path, cls: Any = None, sortkeys: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_pointer:
        json.dump(d, file_pointer, indent=2, cls=cls, sort_keys=sortkeys)
        file_pointer.write("\n")


def dict_to_list(data: Dict[str, List[Any]], keys: List[str]) -> List[Dict[str, Any]]:
    return [{key: data[key][index] for key in keys} for index in range(len(data[keys[0]]))]
