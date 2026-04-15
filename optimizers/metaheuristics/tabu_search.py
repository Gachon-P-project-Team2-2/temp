"""Tabu Search: 최근 이동한 기지국을 일정 기간 금기."""
from __future__ import annotations

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, random_stations


class TabuSearchOptimizer(Optimizer):
    name = "Tabu Search"
    hyperparams = [
        HyperParam("iterations", "int", default=500, min=50, max=5000, step=50,
                   label="iterations (반복 수)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
        HyperParam("tabu_tenure", "int", default=10, min=1, max=100,
                   label="tabu_tenure (금기 기간)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 500, step_size: float = 50.0,
                 tabu_tenure: int = 10) -> OptimizationResult:
        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        tabu_list: dict[int, int] = {}  # station_idx → until iter
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score}]

        for it in range(1, iterations + 1):
            candidates = []
            for _ in range(20):  # 후보 생성 수
                idx = np.random.randint(0, n_stations)
                if idx in tabu_list and tabu_list[idx] > it:
                    continue
                cand = current.copy()
                cand[idx] += np.random.normal(0, step_size, 2)
                clip_stations(cand, problem)
                score = calculate_score(cand, problem)
                candidates.append((score, cand, idx))

            if not candidates:
                history.append({"iter": it, "current_score": current_score, "best_score": best_score})
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen_score, chosen_stations, moved_idx = candidates[0]
            current, current_score = chosen_stations, chosen_score
            tabu_list[moved_idx] = it + tabu_tenure
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
