"""K-Means 기반 최적화 — sklearn 사용. 메타휴리스틱과 다른 범주이므로 별도 파일."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from .metaheuristics._shared import calculate_score


class KMeansOptimizer(Optimizer):
    name = "K-Means"
    hyperparams = [
        HyperParam("n_init", "int", default=10, min=1, max=50,
                   label="n_init (초기화 횟수)"),
        HyperParam("random_state", "int", default=42, min=-1, max=99999,
                   label="random_state (시드, -1=랜덤)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 n_init: int = 10, random_state: int = 42) -> OptimizationResult:
        rs = None if random_state == -1 else random_state
        km = KMeans(n_clusters=n_stations, n_init=n_init, random_state=rs)
        km.fit(problem.X, sample_weight=problem.weights)
        stations = km.cluster_centers_
        score = calculate_score(stations, problem)
        metrics = compute_metrics(stations, problem)
        return OptimizationResult(
            stations=stations,
            score=score,
            metrics=metrics,
            history=None,  # K-Means는 수렴 이력 제공 안함
        )
