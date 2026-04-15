"""Simulated Annealing: 악화 수용을 확률적으로 허용, 온도 감소."""
from __future__ import annotations

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, perturb, random_stations


class SimulatedAnnealingOptimizer(Optimizer):
    name = "Simulated Annealing"
    hyperparams = [
        HyperParam("iterations", "int", default=1000, min=100, max=10000, step=100,
                   label="iterations (반복 수)"),
        HyperParam("initial_temp", "float", default=100.0, min=1.0, max=1000.0, step=1.0,
                   label="initial_temp (초기 온도)"),
        HyperParam("cooling_rate", "float", default=0.99, min=0.80, max=0.999, step=0.001,
                   label="cooling_rate (냉각률)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 1000, initial_temp: float = 100.0,
                 cooling_rate: float = 0.99, step_size: float = 50.0) -> OptimizationResult:
        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        temp = initial_temp
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score, "temp": temp}]

        for it in range(1, iterations + 1):
            next_stations = clip_stations(perturb(current, step_size), problem)
            next_score = calculate_score(next_stations, problem)
            delta = current_score - next_score  # 양수 = 악화
            if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-5)):
                current, current_score = next_stations, next_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            temp *= cooling_rate
            history.append({"iter": it, "current_score": current_score, "best_score": best_score, "temp": temp})

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )
