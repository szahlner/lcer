from typing import Any, Dict, List

from lcer import HER, SAC

ALGOS: Dict[str, Any] = {
    "her": HER,
    "sac": SAC,
}

POLICIES: List[str] = [
    "Gaussian",
    "Deterministic",
]
