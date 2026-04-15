"""합성 트래픽 패턴 생성기 — bs_opt/kmj/core/grid.py에서 포팅.

각 함수는 (rows, cols) 크기의 [0, 1] 정규화된 트래픽 맵을 반환한다.
SyntheticEnvironment는 이 출력에 max_intensity를 곱하고 base_intensity를 더해
기존 스케일(절대값 기반)과 호환되게 변환한다.

패턴 목록:
    random            균일 랜덤
    center_hotspot    중앙 가우시안 + 노이즈
    multi_hotspot     여러 가우시안 합 (bs_simulator 기본 패턴과 호환)
    ring              도넛 모양
    gradient          선형 증가 (동→서 또는 북→남)
    stripe            도로/강 같은 띠
    checkerboard      체커보드 블록
    random_clusters   포아송 클러스터 + 가우시안 커널
"""
from __future__ import annotations

import numpy as np

# UI selectbox에 노출되는 순서와 표시명
PATTERN_CHOICES = [
    "multi_hotspot",
    "center_hotspot",
    "random",
    "ring",
    "gradient",
    "stripe",
    "checkerboard",
    "random_clusters",
]


def _normalize_and_scale(traffic: np.ndarray, params: dict) -> np.ndarray:
    eps = 1e-8
    if params.get("normalize", True):
        traffic = (traffic - traffic.min()) / (traffic.max() - traffic.min() + eps)
    if params.get("clip_0_1", True):
        traffic = np.clip(traffic, 0.0, 1.0)
    return traffic


def _add_noise(traffic: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    noise_std = float(params.get("noise_std", 0.05))
    if noise_std <= 0:
        return traffic
    return traffic + rng.normal(0.0, noise_std, size=traffic.shape)


def generate_pattern(
    rows: int,
    cols: int,
    pattern: str = "multi_hotspot",
    rng: np.random.Generator | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """(rows, cols) 정규화된 트래픽 맵 생성.

    pattern 목록은 PATTERN_CHOICES 참조.
    params는 패턴별 세부 하이퍼파라미터 + 공통 옵션(noise_std, normalize, clip_0_1).
    """
    if rng is None:
        rng = np.random.default_rng()
    params = params or {}

    y, x = np.mgrid[0:rows, 0:cols]
    height, width = rows, cols

    if pattern == "random":
        t = rng.random((height, width))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "center_hotspot":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        sx = params.get("sigma_x", width / 4.0)
        sy = params.get("sigma_y", height / 4.0)
        t = np.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))
        noise_params = dict(params)
        noise_params.setdefault("noise_std", 0.2)
        t = _add_noise(t, rng, noise_params)
        return _normalize_and_scale(t, params)

    if pattern == "multi_hotspot":
        centers = params.get("centers")
        n_centers = params.get("n_centers", 5)
        sx = params.get("sigma_x", width / 6.0)
        sy = params.get("sigma_y", height / 6.0)
        if centers is None:
            centers = [
                (rng.uniform(0, width - 1), rng.uniform(0, height - 1))
                for _ in range(n_centers)
            ]
        t = np.zeros_like(x, dtype=float)
        for cx, cy in centers:
            t += np.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "ring":
        center = params.get("center", ((width - 1) / 2.0, (height - 1) / 2.0))
        radius = float(params.get("radius", min(width, height) / 3.0))
        thickness = float(params.get("thickness", radius / 4.0))
        cx, cy = center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        t = np.exp(-((r - radius) ** 2) / (2 * thickness**2))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "gradient":
        direction = params.get("direction", "ew")
        if direction == "ew":
            base = np.linspace(0, 1, width)[None, :]
            t = np.repeat(base, height, axis=0)
        elif direction == "ns":
            base = np.linspace(0, 1, height)[:, None]
            t = np.repeat(base, width, axis=1)
        else:
            raise ValueError(f"Unknown gradient direction: {direction}")
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "stripe":
        orientation = params.get("orientation", "vertical")
        stripe_pos = params.get(
            "stripe_pos", width // 2 if orientation == "vertical" else height // 2
        )
        stripe_width = params.get("stripe_width", max(2, min(width, height) // 20))
        decay = params.get("decay", stripe_width)
        if orientation == "vertical":
            dist = np.abs(x - stripe_pos)
        elif orientation == "horizontal":
            dist = np.abs(y - stripe_pos)
        else:
            raise ValueError(f"Unknown stripe orientation: {orientation}")
        t = np.exp(-(dist**2) / (2 * (decay**2)))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "checkerboard":
        block = params.get("block", max(2, min(width, height) // 10))
        high = params.get("high", 1.0)
        low = params.get("low", 0.2)
        t = ((x // block + y // block) % 2).astype(float)
        t = t * (high - low) + low
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "random_clusters":
        n_clusters = params.get("n_clusters", int(rng.poisson(5)))
        sigma = params.get("sigma", min(width, height) / 8.0)
        t = np.zeros_like(x, dtype=float)
        for _ in range(max(1, n_clusters)):
            cx = rng.uniform(0, width - 1)
            cy = rng.uniform(0, height - 1)
            t += np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2 * sigma**2))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    raise ValueError(f"Unknown pattern: {pattern!r}. "
                     f"Choices: {PATTERN_CHOICES}")
