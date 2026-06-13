"""Synthetic traffic pattern generators.

Each function returns a normalized traffic map with shape ``(rows, cols)`` and
values in ``[0, 1]`` by default. ``SyntheticEnvironment`` scales the result with
``base_intensity + max_intensity * traffic``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PATTERN_CHOICES = [
    "random_clusters",
    "multi_hotspot",
    "center_hotspot",
    "random",
    "ring",
    "gradient",
    "stripe",
    "checkerboard",
]


def _positive_float(value: Any, default: float, *, min_value: float = 1e-6) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    if not np.isfinite(v) or v < min_value:
        return float(max(default, min_value))
    return v


def _positive_int(value: Any, default: int, *, min_value: int = 1) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = int(default)
    return max(int(min_value), v)


def _normalize_and_scale(traffic: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    traffic = np.asarray(traffic, dtype=float)
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=1.0, neginf=0.0)

    if params.get("normalize", True):
        mn = float(np.min(traffic))
        mx = float(np.max(traffic))
        denom = mx - mn
        if denom <= 1e-12:
            traffic = np.zeros_like(traffic, dtype=float)
        else:
            traffic = (traffic - mn) / denom

    if params.get("clip_0_1", True):
        traffic = np.clip(traffic, 0.0, 1.0)
    return traffic


def _add_noise(traffic: np.ndarray, rng: np.random.Generator, params: dict[str, Any]) -> np.ndarray:
    noise_std = _positive_float(params.get("noise_std", 0.05), 0.05, min_value=0.0)
    if noise_std <= 0:
        return traffic
    return traffic + rng.normal(0.0, noise_std, size=traffic.shape)


def generate_pattern(
    rows: int,
    cols: int,
    pattern: str = "random_clusters",
    rng: np.random.Generator | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Generate a normalized traffic map with shape ``(rows, cols)``.

    Args:
        rows: Number of grid rows. Must be positive.
        cols: Number of grid columns. Must be positive.
        pattern: One of ``PATTERN_CHOICES``.
        rng: Optional NumPy random generator.
        params: Pattern-specific parameters plus common options such as
            ``noise_std``, ``normalize`` and ``clip_0_1``.
    """
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive; got rows={rows}, cols={cols}")

    if rng is None:
        rng = np.random.default_rng()
    params = dict(params or {})

    y, x = np.mgrid[0:rows, 0:cols]
    height, width = rows, cols

    if pattern == "random":
        t = rng.random((height, width))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "center_hotspot":
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        sx = _positive_float(params.get("sigma_x", width / 4.0), max(width / 4.0, 1.0))
        sy = _positive_float(params.get("sigma_y", height / 4.0), max(height / 4.0, 1.0))
        t = np.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))
        noise_params = dict(params)
        noise_params.setdefault("noise_std", 0.2)
        t = _add_noise(t, rng, noise_params)
        return _normalize_and_scale(t, params)

    if pattern == "multi_hotspot":
        centers = params.get("centers")
        n_centers = _positive_int(params.get("n_centers", 5), 5)
        sx = _positive_float(params.get("sigma_x", width / 6.0), max(width / 6.0, 1.0))
        sy = _positive_float(params.get("sigma_y", height / 6.0), max(height / 6.0, 1.0))
        if centers is None:
            centers = [(rng.uniform(0, width - 1), rng.uniform(0, height - 1)) for _ in range(n_centers)]
        t = np.zeros_like(x, dtype=float)
        for cx, cy in centers:
            t += np.exp(-(((x - float(cx)) ** 2) / (2 * sx**2) + ((y - float(cy)) ** 2) / (2 * sy**2)))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "ring":
        center = params.get("center", ((width - 1) / 2.0, (height - 1) / 2.0))
        radius = _positive_float(params.get("radius", min(width, height) / 3.0), max(min(width, height) / 3.0, 1.0))
        thickness = _positive_float(params.get("thickness", radius / 4.0), max(radius / 4.0, 1.0))
        cx, cy = center
        r = np.sqrt((x - float(cx)) ** 2 + (y - float(cy)) ** 2)
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
        stripe_pos = params.get("stripe_pos", width // 2 if orientation == "vertical" else height // 2)
        stripe_width = _positive_int(params.get("stripe_width", max(2, min(width, height) // 20)), 2)
        decay = _positive_float(params.get("decay", stripe_width), float(stripe_width))
        if orientation == "vertical":
            dist = np.abs(x - float(stripe_pos))
        elif orientation == "horizontal":
            dist = np.abs(y - float(stripe_pos))
        else:
            raise ValueError(f"Unknown stripe orientation: {orientation}")
        t = np.exp(-(dist**2) / (2 * decay**2))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "checkerboard":
        block = _positive_int(params.get("block", max(2, min(width, height) // 10)), 2)
        high = float(params.get("high", 1.0))
        low = float(params.get("low", 0.2))
        t = ((x // block + y // block) % 2).astype(float)
        t = t * (high - low) + low
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    if pattern == "random_clusters":
        n_clusters = _positive_int(params.get("n_clusters", int(rng.poisson(5))), 5)
        sigma = _positive_float(params.get("sigma", min(width, height) / 8.0), max(min(width, height) / 8.0, 1.0))
        t = np.zeros_like(x, dtype=float)
        for _ in range(n_clusters):
            cx = rng.uniform(0, width - 1)
            cy = rng.uniform(0, height - 1)
            t += np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / (2 * sigma**2))
        t = _add_noise(t, rng, params)
        return _normalize_and_scale(t, params)

    raise ValueError(f"Unknown pattern: {pattern!r}. Choices: {PATTERN_CHOICES}")
