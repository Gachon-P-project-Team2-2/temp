"""메타휴리스틱 알고리즘들이 공유하는 내부 유틸.

외부(app.py, kmeans.py, rl/)에서 import 금지. 이 폴더 안에서만 사용.
"""
from __future__ import annotations

import numpy as np

from ..base import ProblemInput


def calculate_score(stations: np.ndarray, problem: ProblemInput) -> float:
    """기지국 배치의 점수 (높을수록 좋음).

    score = Σ min(station_load, capacity) + 0.1 · covered_grid_count
    """
    if len(stations) == 0:
        return 0.0

    diff = problem.X[:, np.newaxis, :] - stations[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    radius_sq = problem.radius_m ** 2

    covered_mask = dist_sq <= radius_sq
    is_covered = np.any(covered_mask, axis=1)

    dist_sq_masked = np.where(covered_mask, dist_sq, np.inf)
    nearest_station_idx = np.argmin(dist_sq_masked, axis=1)

    station_loads = np.zeros(len(stations))
    valid_indices = np.where(is_covered)[0]
    if len(valid_indices) == 0:
        return 0.0

    assigned = nearest_station_idx[valid_indices]
    traffic_values = problem.weights[valid_indices]
    np.add.at(station_loads, assigned, traffic_values)

    effective_loads = np.minimum(station_loads, problem.capacity)
    total_covered_traffic = np.sum(effective_loads)
    total_covered_area = len(valid_indices)

    return float(total_covered_traffic + total_covered_area * 0.1)


def random_stations(k: int, problem: ProblemInput) -> np.ndarray:
    """범위 내 균일 랜덤 초기 배치."""
    x = np.random.uniform(0, problem.width_m, k)
    y = np.random.uniform(0, problem.height_m, k)
    return np.column_stack([x, y])


def clip_stations(stations: np.ndarray, problem: ProblemInput) -> np.ndarray:
    """기지국 좌표를 맵 범위 안으로 클립 (in-place)."""
    stations[:, 0] = np.clip(stations[:, 0], 0, problem.width_m)
    stations[:, 1] = np.clip(stations[:, 1], 0, problem.height_m)
    return stations


def perturb(stations: np.ndarray, step_size: float) -> np.ndarray:
    """모든 기지국에 가우시안 노이즈 추가한 새 배열 반환."""
    noise = np.random.normal(0, step_size, size=stations.shape)
    return stations + noise
