"""메타휴리스틱 알고리즘들이 공유하는 내부 유틸.

외부(app.py, kmeans.py, rl/)에서 import 금지. 이 폴더 안에서만 사용.
"""
from __future__ import annotations

import numpy as np

from ..base import ProblemInput, sinr_coverage, spectral_efficiency
from ..base import _resolve_station_params


def calculate_score(stations: np.ndarray, problem: ProblemInput) -> float:
    """기지국 배치의 점수 (높을수록 좋음).

    score_mode='traffic':    Σ min(station_load, capacity)   — 트래픽 커버리지
    score_mode='throughput': Σ bandwidth × η(SINR_i)         — 총 처리량 [Mbps]
    """
    K = len(stations)
    if K == 0:
        return 0.0

    is_covered, serving_idx, best_sinr_db = sinr_coverage(stations, problem)
    _, capacities = _resolve_station_params(problem, K, default_radius=0.0, default_capacity=0.0)

    valid_indices = np.where(is_covered)[0]
    if len(valid_indices) == 0:
        return 0.0

    score_mode = getattr(problem, "score_mode", "traffic")
    if score_mode == "throughput":
        se_mode = getattr(problem, "spectral_efficiency_mode", "shannon")
        eta = spectral_efficiency(best_sinr_db[valid_indices], se_mode)
        station_eta_sum = np.zeros(K)
        station_count = np.zeros(K)
        np.add.at(station_eta_sum, serving_idx[valid_indices], eta)
        np.add.at(station_count, serving_idx[valid_indices], 1.0)
        mean_eta = np.where(station_count > 0, station_eta_sum / station_count, 0.0)
        station_capacity_tp = problem.bandwidth_mhz * mean_eta
        station_demands = np.zeros(K)
        np.add.at(station_demands, serving_idx[valid_indices], problem.weights[valid_indices])
        station_tp = np.minimum(station_capacity_tp, station_demands)
        return float(station_tp.sum())

    # traffic (default)
    station_loads = np.zeros(K)
    np.add.at(station_loads, serving_idx[valid_indices], problem.weights[valid_indices])
    return float(np.sum(np.minimum(station_loads, capacities)))


def random_stations(k: int, problem: ProblemInput) -> np.ndarray:
    """범위 내 균일 랜덤 초기 배치."""
    station_pool = get_station_pool(problem)
    if station_pool is not None:
        if len(station_pool) == 0:
            raise ValueError("기지국을 설치할 수 있는 위치가 없습니다.")
        replace = len(station_pool) < k
        idx = np.random.choice(len(station_pool), size=k, replace=replace)
        return np.asarray(station_pool[idx], dtype=float).copy()
    x = np.random.uniform(0, problem.width_m, k)
    y = np.random.uniform(0, problem.height_m, k)
    return np.column_stack([x, y])


def clip_stations(stations: np.ndarray, problem: ProblemInput) -> np.ndarray:
    """기지국 좌표를 맵 범위 안으로 클립 (in-place)."""
    station_pool = get_station_pool(problem)
    if station_pool is not None:
        if len(station_pool) == 0:
            raise ValueError("기지국을 설치할 수 있는 위치가 없습니다.")
        stations[:] = snap_stations_to_candidates(stations, problem)
        return stations
    stations[:, 0] = np.clip(stations[:, 0], 0, problem.width_m)
    stations[:, 1] = np.clip(stations[:, 1], 0, problem.height_m)
    return stations


def get_station_pool(problem: ProblemInput) -> np.ndarray | None:
    pool = getattr(problem, "feasible_station_points", None)
    if pool is not None:
        return np.asarray(pool, dtype=float)
    pool = getattr(problem, "station_candidate_points", None)
    if pool is not None:
        return np.asarray(pool, dtype=float)
    return None


def snap_stations_to_candidates(stations: np.ndarray, problem: ProblemInput) -> np.ndarray:
    """후보 지점이 있으면 각 기지국 좌표를 가장 가까운 후보 지점으로 이동."""
    candidates = get_station_pool(problem)
    if candidates is None or len(stations) == 0:
        return stations
    if len(candidates) == 0:
        raise ValueError("기지국을 설치할 수 있는 위치가 없습니다.")
    diff = stations[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    nearest = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    return candidates[nearest].copy()


def perturb(stations: np.ndarray, step_size: float) -> np.ndarray:
    """모든 기지국에 가우시안 노이즈 추가한 새 배열 반환."""
    noise = np.random.normal(0, step_size, size=stations.shape)
    return stations + noise
