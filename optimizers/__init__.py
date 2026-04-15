"""Optimizer 플러그인 아키텍처 공개 API.

사용 예:
    from optimizers import REGISTRY, get_optimizer, ProblemInput

    problem = ProblemInput.from_env(env, radius_m=300, capacity=2000)
    optimizer = get_optimizer("K-Means")
    result = optimizer.optimize(problem, n_stations=5, n_init=10, random_state=42)
"""
from __future__ import annotations

from .base import (
    HyperParam,
    Optimizer,
    OptimizationResult,
    ProblemInput,
    compute_metrics,
    convert_to_geo,
)
from .kmeans import KMeansOptimizer
from .metaheuristics.random_walk import RandomWalkOptimizer
from .metaheuristics.simulated_annealing import SimulatedAnnealingOptimizer
from .metaheuristics.tabu_search import TabuSearchOptimizer


# 등록 순서 = UI selectbox 표시 순서
REGISTRY: list[type[Optimizer]] = [
    KMeansOptimizer,
    RandomWalkOptimizer,
    SimulatedAnnealingOptimizer,
    TabuSearchOptimizer,
]


_NAME_TO_CLASS = {cls.name: cls for cls in REGISTRY}


def get_optimizer(name: str) -> Optimizer:
    """이름으로 Optimizer 인스턴스를 반환."""
    if name not in _NAME_TO_CLASS:
        raise ValueError(f"Unknown optimizer: {name!r}. "
                         f"Available: {list(_NAME_TO_CLASS)}")
    return _NAME_TO_CLASS[name]()


def available_names() -> list[str]:
    """UI selectbox용 이름 목록."""
    return [cls.name for cls in REGISTRY]


__all__ = [
    "HyperParam",
    "Optimizer",
    "OptimizationResult",
    "ProblemInput",
    "REGISTRY",
    "get_optimizer",
    "available_names",
    "compute_metrics",
    "convert_to_geo",
]
