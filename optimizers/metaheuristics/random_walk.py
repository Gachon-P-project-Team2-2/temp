"""Random Walk: 랜덤 섭동 후 개선 시 수용, 아니면 버림."""
from __future__ import annotations

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, perturb, random_stations


class RandomWalkOptimizer(Optimizer):
    name = "Random Walk"
    hyperparams = [
        HyperParam("iterations", "int", default=1000, min=100, max=10000, step=100,
                   label="iterations (반복 수)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 1000, step_size: float = 50.0) -> OptimizationResult:
        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score}]

        for it in range(1, iterations + 1):
            next_stations = clip_stations(perturb(current, step_size), problem)
            next_score = calculate_score(next_stations, problem)
            if next_score > current_score:
                current, current_score = next_stations, next_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            history.append({"iter": it, "current_score": current_score, "best_score": best_score})

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )
