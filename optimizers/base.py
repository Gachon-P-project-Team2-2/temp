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
# MCS 테이블 — 3GPP LTE CQI 유사 (SINR dB 하한 → 스펙트럼 효율 bits/s/Hz)
# ---------------------------------------------------------------------------
_MCS_SINR_DB  = np.array([-6, -4, -2,  0,  2,  4,  6,   8,  10,  14,  18,  22,  26], dtype=float)
_MCS_SPECTRAL = np.array([0.0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18,
                           1.48, 1.91, 2.73, 3.32, 3.90, 4.52], dtype=float)
_MCS_MAX_EFF  = 4.80  # 256QAM 최고 효율 (SINR ≥ 26 dB)


def capacity_from_bandwidth(bandwidth_mhz: float, overhead_ratio: float = 0.15) -> float:
    """대역폭과 오버헤드로부터 기지국 피크 처리 용량 [Mbps] 유도.

    C = bandwidth_mhz × η_max × (1 - overhead_ratio)
    η_max = 4.80 bits/s/Hz (256QAM 이론 최대)
    """
    return float(bandwidth_mhz) * _MCS_MAX_EFF * (1.0 - float(overhead_ratio))


def spectral_efficiency(sinr_db: np.ndarray, mode: str = "shannon") -> np.ndarray:
    """SINR(dB) → 스펙트럼 효율 [bits/s/Hz].

    mode='shannon': 이론 상한  log₂(1 + 10^(SINR/10))
    mode='mcs':     3GPP LTE 유사 이산 테이블
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    if mode == "mcs":
        idx = np.searchsorted(_MCS_SINR_DB, sinr_db, side="right") - 1
        idx = np.clip(idx, 0, len(_MCS_SPECTRAL) - 1)
        result = _MCS_SPECTRAL[idx].copy()
        result[sinr_db >= 26.0] = _MCS_MAX_EFF
        return result
    # shannon (default)
    return np.log2(1.0 + 10.0 ** (sinr_db / 10.0))


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
    radius_m: float | np.ndarray
    capacity: float | np.ndarray

    # Geo 변환용 (stations → lat/lon 렌더링에 사용)
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    station_candidate_points: np.ndarray | None = None
    feasible_station_points: np.ndarray | None = None

    # ---- 전파 모델 (log-distance 경로 손실 + SINR 기반 커버리지) ----
    path_loss_exponent: float = 3.5           # 경로 손실 지수 n (자유공간=2, 도심=3.5)
    path_loss_ref_db: float = 38.0            # d=1m 기준 경로 손실 [dB] (~1.8 GHz)
    tx_power_dbm: float | np.ndarray = 43.0   # 기지국 송신 전력 [dBm] (배열: 기지국별 HetNet)
    noise_floor_dbm: float = -97.0            # 잡음 바닥 [dBm] (10 MHz BW, NF 7 dB 포함)
    sinr_threshold_db: float = 3.0            # 커버리지 최소 SINR [dB]
    bandwidth_mhz: float = 10.0               # 시스템 대역폭 [MHz]

    # ---- 최적화 목표 ----
    score_mode: str = "traffic"               # "traffic" | "throughput"
    spectral_efficiency_mode: str = "shannon" # "shannon" | "mcs"

    # ---- 트래픽 단위 변환 ----
    weight_scale: float = 1.0                 # weights → Mbps 스케일 팩터

    @classmethod
    def from_env(
        cls,
        env,
        radius_m: float | np.ndarray | list[float] | tuple[float, ...],
        capacity: float | np.ndarray | list[float] | tuple[float, ...],
        station_candidate_points=None,
        *,
        path_loss_exponent: float = 3.5,
        path_loss_ref_db: float = 38.0,
        tx_power_dbm: float | np.ndarray = 43.0,
        noise_floor_dbm: float = -97.0,
        sinr_threshold_db: float = 3.0,
        bandwidth_mhz: float = 10.0,
        score_mode: str = "traffic",
        spectral_efficiency_mode: str = "shannon",
        weight_scale: float = 1.0,
    ) -> "ProblemInput":
        """SyntheticEnvironment 인스턴스에서 ProblemInput 구축."""
        data = env.get_local_data()
        if data.size == 0:
            data = np.empty((0, 3))
        candidates = station_candidate_points
        if candidates is None:
            candidates = getattr(env, "station_candidate_points", None)
        candidate_array = _normalize_candidate_points(candidates)
        has_candidate_constraint = candidate_array is not None
        if candidate_array is not None and hasattr(env, "filter_station_candidate_points"):
            candidate_array = env.filter_station_candidate_points(candidate_array)
        feasible_array = _build_feasible_station_points(env, candidate_array, has_candidate_constraint)
        weights = data[:, 2] * float(weight_scale) if weight_scale != 1.0 else data[:, 2]
        return cls(
            X=data[:, 0:2],
            weights=weights,
            width_m=env.width_m,
            height_m=env.height_m,
            radius_m=radius_m,
            capacity=capacity,
            lat_min=env.lat_min, lat_max=env.lat_max,
            lon_min=env.lon_min, lon_max=env.lon_max,
            station_candidate_points=candidate_array,
            feasible_station_points=feasible_array,
            path_loss_exponent=path_loss_exponent,
            path_loss_ref_db=path_loss_ref_db,
            tx_power_dbm=tx_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            sinr_threshold_db=sinr_threshold_db,
            bandwidth_mhz=bandwidth_mhz,
            score_mode=score_mode,
            spectral_efficiency_mode=spectral_efficiency_mode,
            weight_scale=weight_scale,
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


def _normalize_candidate_points(points) -> np.ndarray | None:
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1 and arr.shape == (2,):
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr.copy()


def _build_feasible_station_points(env, candidate_array: np.ndarray | None,
                                   has_candidate_constraint: bool) -> np.ndarray | None:
    if has_candidate_constraint:
        if candidate_array is None:
            return np.empty((0, 2))
        return candidate_array.copy()
    if hasattr(env, "get_station_feasible_points"):
        feasible = env.get_station_feasible_points()
        if feasible is not None:
            return np.asarray(feasible, dtype=float).copy()
    return None


def _as_station_array(values: float | np.ndarray | list[float] | tuple[float, ...], n: int,
                     default: float | None = None) -> np.ndarray:
    """스칼라/배열 입력을 n개 스펙 배열로 정규화한다."""
    if default is None:
        default = 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full(n, float(default), dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if n <= 0:
        return np.array([], dtype=float)
    if arr.size < n:
        pad = np.full(n - arr.size, float(arr[-1]), dtype=float)
        return np.concatenate([arr.astype(float), pad], axis=0)
    return arr[:n].astype(float)


def _resolve_station_params(problem: ProblemInput, n_stations: int, default_radius: float = 0.0,
                           default_capacity: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """문제 인스턴스에서 n_stations에 맞는 반경/용량 배열을 반환한다."""
    radius = _as_station_array(problem.radius_m, n_stations, default=default_radius)
    capacity = _as_station_array(problem.capacity, n_stations, default=default_capacity)
    return radius, capacity


def compute_sinr(stations: np.ndarray, problem: ProblemInput) -> np.ndarray:
    """각 셀-기지국 쌍의 SINR (linear scale). 반환 shape = (N, K).

    경로 손실 모델: PL(d) = path_loss_ref_db + 10·n·log10(d/1m)  [dB]
    간섭 모델: SINR_ij = P_rx(i,j) / (N0 + Σ_{k≠j} P_rx(i,k))
    """
    K = len(stations)
    N = len(problem.X)
    if K == 0 or N == 0:
        return np.zeros((N, max(K, 1)))

    diff = problem.X[:, np.newaxis, :] - stations[np.newaxis, :, :]   # (N, K, 2)
    dist_m = np.sqrt(np.sum(diff ** 2, axis=2))                        # (N, K)
    dist_m = np.maximum(dist_m, 1.0)                                    # 최소 기준 거리 1m

    pl_db = problem.path_loss_ref_db + 10.0 * problem.path_loss_exponent * np.log10(dist_m)

    # 기지국별 송신 전력 → 길이 K 배열 정규화
    tx_dbm = np.asarray(problem.tx_power_dbm, dtype=float).ravel()
    if tx_dbm.size == 1:
        tx_dbm = np.full(K, tx_dbm[0])
    elif tx_dbm.size < K:
        tx_dbm = np.concatenate([tx_dbm, np.full(K - tx_dbm.size, tx_dbm[-1])])
    tx_dbm = tx_dbm[:K]

    # 수신 전력 [W]: P_W = 10^((P_dBm − PL − 30) / 10)
    rx_w = 10.0 ** ((tx_dbm[np.newaxis, :] - pl_db - 30.0) / 10.0)    # (N, K)
    noise_w = 10.0 ** ((problem.noise_floor_dbm - 30.0) / 10.0)

    # 기지국 j가 셀 i를 서비스할 때, 나머지 기지국은 간섭원
    total_rx = rx_w.sum(axis=1, keepdims=True)       # (N, 1)
    interference_w = total_rx - rx_w                  # (N, K)
    sinr = rx_w / (noise_w + interference_w + 1e-30)  # (N, K), linear

    return sinr


def sinr_coverage(
    stations: np.ndarray, problem: ProblemInput
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SINR 기반 커버리지의 단일 진실 소스.

    compute_metrics / calculate_score / app.py 시각화 모두 이 함수를 공유한다.

    Returns:
        is_covered   : (N,) bool   — SINR ≥ sinr_threshold_db 인 셀
        serving_idx  : (N,) int    — best-SINR 기지국 인덱스
        best_sinr_db : (N,) float  — 각 셀의 best SINR [dB]
    """
    K = len(stations)
    N = len(problem.X)
    if K == 0:
        return (
            np.zeros(N, dtype=bool),
            np.zeros(N, dtype=int),
            np.full(N, -np.inf),
        )

    sinr = compute_sinr(stations, problem)                       # (N, K)
    threshold = 10.0 ** (problem.sinr_threshold_db / 10.0)

    serving_idx = np.argmax(sinr, axis=1)                        # (N,)
    best_sinr_linear = sinr[np.arange(N), serving_idx]           # (N,)
    is_covered = best_sinr_linear >= threshold                    # (N,)
    best_sinr_db = 10.0 * np.log10(np.maximum(best_sinr_linear, 1e-30))

    return is_covered, serving_idx, best_sinr_db


def compute_metrics(stations: np.ndarray, problem: ProblemInput) -> dict:
    """기지국 배치에 대한 통계.

    SINR 기반 커버리지 + best-SINR 배정.
    spectral_efficiency_mode에 따라 Shannon 또는 MCS 처리량을 보조 메트릭으로 포함.
    """
    K = len(stations)
    se_mode = getattr(problem, "spectral_efficiency_mode", "shannon")
    if K == 0:
        return {
            "total_traffic": float(np.sum(problem.weights)),
            "covered_traffic": 0.0,
            "total_area": int(len(problem.weights)),
            "covered_area": 0,
            "station_loads": np.zeros(0),
            "station_effective_loads": np.zeros(0),
            "capacity": np.array([], dtype=float),
            "mean_sinr_db": None,
            "station_throughput_mbps": np.zeros(0),
            "total_throughput_mbps": 0.0,
        }

    is_covered, serving_idx, best_sinr_db = sinr_coverage(stations, problem)
    _, capacities = _resolve_station_params(problem, K, default_radius=0.0, default_capacity=0.0)

    valid_indices = np.where(is_covered)[0]
    station_loads = np.zeros(K)
    if len(valid_indices) > 0:
        np.add.at(station_loads, serving_idx[valid_indices], problem.weights[valid_indices])
    effective_loads = np.minimum(station_loads, capacities)

    # 처리량 계산: 기지국당 평균 스펙트럼 효율 × 대역폭 (대역폭은 셀 수와 무관하게 공유)
    eta = spectral_efficiency(best_sinr_db, se_mode)  # (N,)
    station_tp = np.zeros(K)
    if len(valid_indices) > 0:
        station_eta_sum = np.zeros(K)
        station_count = np.zeros(K)
        np.add.at(station_eta_sum, serving_idx[valid_indices], eta[valid_indices])
        np.add.at(station_count, serving_idx[valid_indices], 1.0)
        mean_eta = np.where(station_count > 0, station_eta_sum / station_count, 0.0)
        station_capacity_tp = problem.bandwidth_mhz * mean_eta  # Mbps
        station_demands = np.zeros(K)
        np.add.at(station_demands, serving_idx[valid_indices], problem.weights[valid_indices])
        station_tp = np.minimum(station_capacity_tp, station_demands)

    mean_sinr_db = float(np.mean(best_sinr_db[is_covered])) if np.any(is_covered) else None

    return {
        "total_traffic": float(np.sum(problem.weights)),
        "covered_traffic": float(np.sum(effective_loads)),
        "total_area": int(len(problem.weights)),
        "covered_area": int(len(valid_indices)),
        "station_loads": station_loads,
        "station_effective_loads": effective_loads,
        "capacity": capacities,
        "mean_sinr_db": mean_sinr_db,
        "station_throughput_mbps": station_tp,
        "total_throughput_mbps": float(station_tp.sum()),
        # 하위 호환성: 이전 키 유지
        "station_shannon_mbps": station_tp,
    }
