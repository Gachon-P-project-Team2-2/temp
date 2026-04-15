"""Optimizer 플러그인 아키텍처의 기본 계약.

UI와 알고리즘 간 유일한 접점은 이 모듈에서 정의된 3개 타입:
    - ProblemInput:   입력 문제 (읽기 전용)
    - OptimizationResult: 출력 해 + 메트릭
    - Optimizer:      알고리즘이 구현하는 ABC

각 알고리즘은 내부 구조를 자유롭게 선택한다. UI는 내부를 모른다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


# ---------------------------------------------------------------------------
# 하이퍼파라미터 스키마 (UI 위젯 자동 생성용)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HyperParam:
    name: str
    kind: Literal["int", "float", "choice", "bool"]
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    choices: list | None = None
    label: str | None = None  # 없으면 name 사용
    help: str | None = None


# ---------------------------------------------------------------------------
# 문제 입력 / 결과 출력
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemInput:
    """UI → Optimizer 로 전달되는 문제 인스턴스. 불변."""
    X: np.ndarray              # (N, 2) Local 좌표 (x, y in m). 트래픽 > 0인 셀만.
    weights: np.ndarray        # (N,) 각 셀의 트래픽 값
    width_m: float
    height_m: float
    radius_m: float
    capacity: float

    # Geo 변환용 (stations → lat/lon 렌더링에 사용)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_env(cls, env, radius_m: float, capacity: float) -> "ProblemInput":
        """SyntheticEnvironment 인스턴스에서 ProblemInput 구축."""
        data = env.get_local_data()
        return cls(
            X=data[:, 0:2],
            weights=data[:, 2],
            width_m=env.width_m,
            height_m=env.height_m,
            radius_m=radius_m,
            capacity=capacity,
            lat_min=env.lat_min, lat_max=env.lat_max,
            lon_min=env.lon_min, lon_max=env.lon_max,
        )


@dataclass
class OptimizationResult:
    """Optimizer → UI 로 반환되는 결과."""
    stations: np.ndarray       # (k, 2) Local 좌표
    score: float               # 높을수록 좋음 (score↑ 규약)
    metrics: dict              # total_traffic, covered_traffic, station_loads, ...
    history: list[dict] | None = None   # 수렴 이력 (선택)


# ---------------------------------------------------------------------------
# Optimizer ABC
# ---------------------------------------------------------------------------
class Optimizer(ABC):
    """모든 최적화 알고리즘이 구현하는 단일 계약.

    하위 클래스는 name, hyperparams 두 클래스 속성과 optimize 메소드를 구현한다.
    """

    # UI selectbox에 표시되는 이름 (반드시 override)
    name: str = ""

    # UI가 자동으로 위젯을 생성하기 위한 스키마 (반드시 override)
    hyperparams: list[HyperParam] = []

    @abstractmethod
    def optimize(self, problem: ProblemInput, n_stations: int, **hp) -> OptimizationResult:
        """핵심 인터페이스. 하위 클래스에서 구현.

        Args:
            problem: 문제 인스턴스 (읽기 전용으로 취급할 것)
            n_stations: 배치할 기지국 개수
            **hp: hyperparams 스키마에 따라 전달되는 값들

        Returns:
            OptimizationResult — 기지국 위치, 점수, 메트릭
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 공용 유틸: Local → Geo 변환, 통계 계산
# ---------------------------------------------------------------------------
def convert_to_geo(stations_local: np.ndarray, problem: ProblemInput) -> np.ndarray:
    """(k, 2) Local(m) → (k, 2) [lat, lon] 변환."""
    if len(stations_local) == 0:
        return np.empty((0, 2))
    x_scale = (problem.lon_max - problem.lon_min) / problem.width_m
    y_scale = (problem.lat_max - problem.lat_min) / problem.height_m
    lon = problem.lon_min + stations_local[:, 0] * x_scale
    lat = problem.lat_min + stations_local[:, 1] * y_scale
    return np.column_stack([lat, lon])


def compute_metrics(stations: np.ndarray, problem: ProblemInput) -> dict:
    """기지국 배치에 대한 통계 (점수와 별개로 metrics 계산).

    기존 BaseStationOptimizer.get_stats와 동일한 결과를 반환한다.
    """
    if len(stations) == 0:
        return {
            "total_traffic": float(np.sum(problem.weights)),
            "covered_traffic": 0.0,
            "total_area": int(len(problem.weights)),
            "covered_area": 0,
            "station_loads": np.zeros(0),
            "station_effective_loads": np.zeros(0),
            "capacity": problem.capacity,
        }

    diff = problem.X[:, np.newaxis, :] - stations[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    radius_sq = problem.radius_m ** 2

    covered_mask = dist_sq <= radius_sq
    is_covered = np.any(covered_mask, axis=1)

    dist_sq_masked = np.where(covered_mask, dist_sq, np.inf)
    nearest_station_idx = np.argmin(dist_sq_masked, axis=1)

    station_loads = np.zeros(len(stations))
    valid_indices = np.where(is_covered)[0]
    if len(valid_indices) > 0:
        assigned = nearest_station_idx[valid_indices]
        traffic_values = problem.weights[valid_indices]
        np.add.at(station_loads, assigned, traffic_values)

    effective_loads = np.minimum(station_loads, problem.capacity)

    return {
        "total_traffic": float(np.sum(problem.weights)),
        "covered_traffic": float(np.sum(effective_loads)),
        "total_area": int(len(problem.weights)),
        "covered_area": int(len(valid_indices)),
        "station_loads": station_loads,
        "station_effective_loads": effective_loads,
        "capacity": problem.capacity,
    }
