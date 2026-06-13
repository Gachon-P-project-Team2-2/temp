"""
Dash + dash-leaflet version of the base-station placement simulator.

Run:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import logging.handlers
import os
import threading
import time
import traceback
import uuid
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import (
    ALL,
    MATCH,
    Dash,
    Input,
    Output,
    State,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate
import dash_leaflet as dl
from dash_extensions.javascript import assign
from geopy.distance import geodesic

from environment import SyntheticEnvironment, TIME_PROFILES
from obstacle_sources import (
    filter_polygons,
    geojson_to_polygons,
    load_osm_polygons_with_cache,
)
from optimizers import (
    REGISTRY,
    ProblemInput,
    capacity_from_bandwidth,
    compute_metrics,
    compute_sinr,
    convert_to_geo,
    get_optimizer,
    sinr_coverage,
)
from patterns import PATTERN_CHOICES


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_optimizer_logging() -> logging.Logger:
    """optimizers 패키지 전용 로거: logs/optimizer.log (rotating) + stderr."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("optimizers")
    if logger.handlers:          # 재진입(reload) 방지
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 파일 핸들러: 최대 2 MB, 백업 3개
    fh = logging.handlers.RotatingFileHandler(
        "logs/optimizer.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 콘솔 핸들러: WARNING 이상만
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

_opt_logger = _setup_optimizer_logging()

STATION_PIN_MARKER_ENABLED = True


# ---------------------------------------------------------------------------
# Constants / server state
# ---------------------------------------------------------------------------

DEFAULT_CENTER = [37.4979, 127.0276]
DEFAULT_ZOOM = 14

OSM_OBSTACLE_TYPE_LABELS = ["건물", "수역/물길", "도로"]
OSM_OBSTACLE_TYPE_VALUES = {
    "건물": "building",
    "수역/물길": ("water", "waterway"),
    "도로": "road",
}
OSM_OBJECT_USAGE_MODES = ["장애물로 사용"]
DEFAULT_DYNAMIC_TRAFFIC_TYPE = "moving_hotspot"
DYNAMIC_TRAFFIC_TYPE_OPTIONS = [
    {"label": "고정 위치 변동", "value": "fixed_variation"},
    {"label": "이동형 핫스팟", "value": "moving_hotspot"},
    {"label": "위치 전환형", "value": "switching_locations"},
]
DEFAULT_OPERATION_POLICY = "always-on"
OPERATION_POLICY_OPTIONS = [
    {"label": "always-on", "value": "always-on"},
    {"label": "threshold", "value": "threshold"},
    {"label": "two-threshold", "value": "two-threshold"},
    {"label": "greedy-off", "value": "greedy-off"},
    {"label": "dqn", "value": "dqn"},
]
OPERATION_POLICY_VALUES = {item["value"] for item in OPERATION_POLICY_OPTIONS}
OPERATION_ACTIVE_POWER_TX_MULTIPLIER = 8.0
OPERATION_SLEEP_POWER_TX_MULTIPLIER = 0.5
OPERATION_COST_PARAMS = {
    "load_power_w": 216.0,
    "switching_cost": 5.0,
    "uncovered_penalty": 10.0,
    "overload_penalty": 20.0,
    "sleep_threshold_mbps": 5.0,
}
OPERATION_DEFAULT_PARAMS = {
    **OPERATION_COST_PARAMS,
    "wake_threshold_multiplier": 2.0,
    "dqn_lr": 0.01,
    "dqn_gamma": 0.9,
    "dqn_epsilon": 1.0,
    "dqn_epsilon_decay": 0.95,
    "dqn_epsilon_min": 0.01,
}
OPERATION_PARAM_SPECS = {
    "load_power_w": {"label": "부하 전력 계수 (W)", "step": 1.0, "min": 0.0},
    "switching_cost": {"label": "전환 비용", "step": 0.5, "min": 0.0},
    "uncovered_penalty": {"label": "미커버 페널티", "step": 0.5, "min": 0.0},
    "overload_penalty": {"label": "과부하 페널티", "step": 0.5, "min": 0.0},
    "sleep_threshold_mbps": {"label": "Sleep 임계값 (Mbps)", "step": 0.5, "min": 0.0},
    "wake_threshold_multiplier": {"label": "Wake 배수", "step": 0.1, "min": 1.0},
    "dqn_lr": {"label": "DQN 학습률", "step": 0.001, "min": 0.0},
    "dqn_gamma": {"label": "DQN 감가율 γ", "step": 0.01, "min": 0.0, "max": 1.0},
    "dqn_epsilon": {"label": "DQN 초기 epsilon", "step": 0.01, "min": 0.0, "max": 1.0},
    "dqn_epsilon_decay": {"label": "DQN epsilon decay", "step": 0.01, "min": 0.0, "max": 1.0},
    "dqn_epsilon_min": {"label": "DQN 최소 epsilon", "step": 0.01, "min": 0.0, "max": 1.0},
}
OPERATION_COMMON_PARAM_NAMES = [
    "load_power_w",
    "switching_cost",
    "uncovered_penalty",
    "overload_penalty",
]
OPERATION_POLICY_PARAM_NAMES = {
    "always-on": [],
    "threshold": ["sleep_threshold_mbps"],
    "two-threshold": ["sleep_threshold_mbps", "wake_threshold_multiplier"],
    "greedy-off": [],
    "dqn": ["dqn_lr", "dqn_gamma", "dqn_epsilon", "dqn_epsilon_decay", "dqn_epsilon_min"],
}

APP_STATE: dict[str, dict[str, Any]] = {}
_LAST_ACCESSED: dict[str, float] = {}
_SESSION_TTL = 3_600.0  # 1시간 미접근 세션 자동 삭제
_APP_STATE_LOCK = threading.Lock()


TRAFFIC_STYLE = assign(
    """
function(feature, context){
    return {
        fillColor: feature.properties.fillColor || "#ff0000",
        color: "transparent",
        weight: 0,
        fillOpacity: feature.properties.fillOpacity || 0.2,
        interactive: feature.properties.interactive || false
    };
}
"""
)

TRAFFIC_ON_EACH_FEATURE = assign(
    """
function(feature, layer, context){
    var p = feature.properties || {};
    var sinrStr = (p.sinr_db != null) ? (p.sinr_db.toFixed(1) + " dB") : "N/A";
    layer.bindTooltip(
        "Traffic: " + (p.traffic ?? "-") +
        "<br>Status: " + (p.status ?? "-") +
        "<br>SINR: " + sinrStr +
        "<br>Area: " + (p.obstacle ?? "-"),
        {sticky: true}
    );
}
"""
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def get_session_state(session_id: str) -> dict[str, Any]:
    if not session_id:
        raise PreventUpdate
    now = time.time()
    with _APP_STATE_LOCK:
        _LAST_ACCESSED[session_id] = now
        if len(APP_STATE) > 50:
            stale = [sid for sid, ts in list(_LAST_ACCESSED.items()) if now - ts > _SESSION_TTL]
            for sid in stale:
                APP_STATE.pop(sid, None)
                _LAST_ACCESSED.pop(sid, None)
        return APP_STATE.setdefault(session_id, {})


def version_token() -> dict[str, Any]:
    return {"version": time.time()}


def normalize_triggered_bool(value: Any) -> bool:
    return bool(value)


def normalize_operation_policy(value: Any) -> str:
    value = str(value or DEFAULT_OPERATION_POLICY)
    return value if value in OPERATION_POLICY_VALUES else DEFAULT_OPERATION_POLICY


def decode_upload_to_bytes(contents: str | None) -> io.BytesIO | None:
    if not contents:
        return None
    try:
        _, encoded = contents.split(",", 1)
        return io.BytesIO(base64.b64decode(encoded))
    except Exception:
        return None


def safe_float(value: Any, default: float) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_operation_params(values: dict[str, Any] | None = None) -> dict[str, float]:
    source = values or {}
    params = {
        name: safe_float(source.get(name), default)
        for name, default in OPERATION_DEFAULT_PARAMS.items()
    }

    for name in (
        "load_power_w",
        "switching_cost",
        "uncovered_penalty",
        "overload_penalty",
        "sleep_threshold_mbps",
    ):
        params[name] = max(0.0, params[name])
    params["wake_threshold_multiplier"] = max(1.0, params["wake_threshold_multiplier"])
    params["dqn_lr"] = max(0.0, params["dqn_lr"])
    params["dqn_gamma"] = min(1.0, max(0.0, params["dqn_gamma"]))
    params["dqn_epsilon"] = min(1.0, max(0.0, params["dqn_epsilon"]))
    params["dqn_epsilon_decay"] = min(1.0, max(0.0, params["dqn_epsilon_decay"]))
    params["dqn_epsilon_min"] = min(1.0, max(0.0, params["dqn_epsilon_min"]))
    return params


def operation_param_names_for_policy(policy: str) -> list[str]:
    policy = normalize_operation_policy(policy)
    return OPERATION_COMMON_PARAM_NAMES + OPERATION_POLICY_PARAM_NAMES.get(policy, [])


def operation_active_mask_for_frame(
    operation_results: dict[str, Any] | None,
    station_count: int,
    frame_index: int,
) -> np.ndarray | None:
    history = (operation_results or {}).get("history") or []
    if station_count <= 0 or not history:
        return None

    idx = max(0, min(safe_int(frame_index, 0), len(history) - 1))
    active_mask = history[idx].get("active_mask")
    if not isinstance(active_mask, (list, tuple)):
        return None

    mask = np.asarray(active_mask, dtype=bool)
    if len(mask) != station_count:
        return None
    return mask


def _parse_operation_hyperparams(param_values, param_ids) -> dict[str, float]:
    values: dict[str, Any] = {}
    for value, id_obj in zip(param_values or [], param_ids or []):
        name = id_obj.get("name") if isinstance(id_obj, dict) else None
        if name in OPERATION_DEFAULT_PARAMS:
            values[name] = value
    return normalize_operation_params(values)


def has_custom_region(custom_region: Any) -> bool:
    return (
        isinstance(custom_region, dict)
        and bool(custom_region.get("width_km"))
        and bool(custom_region.get("height_km"))
    )


def parse_map_center(center: Any) -> tuple[float, float]:
    if isinstance(center, dict):
        lat = center.get("lat", center.get("latitude"))
        lon = center.get("lng", center.get("lon", center.get("longitude")))
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])

    return float(DEFAULT_CENTER[0]), float(DEFAULT_CENTER[1])


def parse_map_bounds(bounds: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not bounds:
        return None

    if isinstance(bounds, dict):
        sw = bounds.get("_southWest") or bounds.get("southWest")
        ne = bounds.get("_northEast") or bounds.get("northEast")
        if isinstance(sw, dict) and isinstance(ne, dict):
            sw_lat = sw.get("lat")
            sw_lng = sw.get("lng", sw.get("lon"))
            ne_lat = ne.get("lat")
            ne_lng = ne.get("lng", ne.get("lon"))
            if None not in (sw_lat, sw_lng, ne_lat, ne_lng):
                return (float(sw_lat), float(sw_lng)), (float(ne_lat), float(ne_lng))

    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        sw = bounds[0]
        ne = bounds[1]
        if isinstance(sw, (list, tuple)) and isinstance(ne, (list, tuple)) and len(sw) >= 2 and len(ne) >= 2:
            return (float(sw[0]), float(sw[1])), (float(ne[0]), float(ne[1]))

    return None


def ensure_station_spec_rows(
    rows: list[dict[str, Any]] | None,
    target_count: int,
    default_radius: float,
    default_capacity: float,
    default_tx_power: float = 43.0,
    default_bandwidth: float = 10.0,
) -> list[dict[str, Any]]:
    rows = rows or []
    norm_rows: list[dict[str, Any]] = []

    for i in range(max(0, int(target_count))):
        old = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
        norm_rows.append(
            {
                "station": i + 1,
                "radius_m": safe_float(old.get("radius_m"), default_radius),
                "capacity": safe_float(old.get("capacity"), default_capacity),
                "tx_power_dbm": safe_float(old.get("tx_power_dbm"), default_tx_power),
                "bandwidth_mhz": safe_float(old.get("bandwidth_mhz"), default_bandwidth),
            }
        )

    return norm_rows


def coerce_station_tx_power_array(
    rows: list[dict[str, Any]] | None,
    station_points: int,
    fallback_tx: float,
) -> np.ndarray:
    rows = rows or []
    tx_power = []

    for i in range(station_points):
        if i < len(rows) and isinstance(rows[i], dict):
            tx_power.append(safe_float(rows[i].get("tx_power_dbm"), fallback_tx))
        else:
            tx_power.append(float(fallback_tx))

    return np.asarray(tx_power, dtype=float)


def coerce_station_bandwidth_array(
    rows: list[dict[str, Any]] | None,
    station_points: int,
    fallback_bandwidth: float,
) -> np.ndarray:
    rows = rows or []
    bw = []

    for i in range(station_points):
        if i < len(rows) and isinstance(rows[i], dict):
            bw.append(safe_float(rows[i].get("bandwidth_mhz"), fallback_bandwidth))
        else:
            bw.append(float(fallback_bandwidth))

    return np.asarray(bw, dtype=float)


def coerce_station_capacity_array(
    rows: list[dict[str, Any]] | None,
    station_points: int,
    fallback_capacity: float,
) -> np.ndarray:
    rows = rows or []
    capacity = []

    for i in range(station_points):
        if i < len(rows) and isinstance(rows[i], dict):
            capacity.append(safe_float(rows[i].get("capacity"), fallback_capacity))
        else:
            capacity.append(float(fallback_capacity))

    return np.asarray(capacity, dtype=float)



def tx_power_for_k(
    k: int,
    ui_tx_power: float,
    spec_mode: str,
    spec_rows: list[dict[str, Any]] | None,
) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=float)

    if spec_mode == "기지국별 개별" and spec_rows:
        base = coerce_station_tx_power_array(spec_rows, k, float(ui_tx_power))
    else:
        base = np.asarray([float(ui_tx_power)], dtype=float)

    if len(base) < k:
        base = np.concatenate([base, np.full(k - len(base), float(base[-1]), dtype=float)])

    return base[:k].astype(float)


def capacity_for_k(
    k: int,
    spec_mode: str,
    spec_rows: list[dict[str, Any]] | None,
    capacity_default: float,
) -> np.ndarray:
    if spec_mode == "기지국별 개별" and spec_rows:
        return coerce_station_capacity_array(spec_rows, k, float(capacity_default))
    return np.full(k, float(capacity_default), dtype=float)


def prop_params_base(
    path_loss_exponent: float,
    bandwidth_mhz: float,
    sinr_threshold_db: float,
    max_coord_stations: int = 1,
) -> dict[str, float]:
    bandwidth_mhz = max(float(bandwidth_mhz), 1e-9)
    noise_floor_dbm = -174.0 + 10.0 * np.log10(bandwidth_mhz * 1e6) + 7.0

    return {
        "path_loss_exponent": float(path_loss_exponent),
        "path_loss_ref_db": 38.0,
        "noise_floor_dbm": float(noise_floor_dbm),
        "sinr_threshold_db": float(sinr_threshold_db),
        "bandwidth_mhz": float(bandwidth_mhz),
        "max_coord_stations": int(max_coord_stations),
    }


def live_visualization_state(
    opt_results: dict[str, Any] | None,
    spec_mode: str | None,
    station_specs: list[dict[str, Any]] | None,
    ui_tx_power: Any,
    ui_path_loss_exp: Any,
    ui_bandwidth_mhz: Any,
    ui_sinr_threshold: Any,
    ui_max_coord: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    if not opt_results:
        return opt_results, station_specs

    prop = dict(opt_results.get("prop_params", {}))
    bandwidth = safe_float(ui_bandwidth_mhz, prop.get("bandwidth_mhz", 10.0))
    live_prop = prop_params_base(
        path_loss_exponent=safe_float(ui_path_loss_exp, prop.get("path_loss_exponent", 3.5)),
        bandwidth_mhz=bandwidth,
        sinr_threshold_db=safe_float(ui_sinr_threshold, prop.get("sinr_threshold_db", 3.0)),
        max_coord_stations=safe_int(ui_max_coord, int(prop.get("max_coord_stations", 1))),
    )
    live_prop["path_loss_ref_db"] = safe_float(prop.get("path_loss_ref_db"), live_prop["path_loss_ref_db"])

    stations = opt_results.get("stations_geo") or []
    station_count = len(stations)
    prop_tx = np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()
    fallback_tx = safe_float(ui_tx_power, float(prop_tx[0]) if len(prop_tx) else 43.0)

    if spec_mode == "기지국별 개별":
        live_specs = station_specs
        tx = coerce_station_tx_power_array(live_specs, station_count, fallback_tx)
    else:
        capacity = capacity_from_bandwidth(bandwidth)
        live_specs = [
            {
                "tx_power_dbm": fallback_tx,
                "bandwidth_mhz": bandwidth,
                "capacity": capacity,
            }
            for _ in range(station_count)
        ]
        tx = np.full(station_count, fallback_tx, dtype=float)

    live_prop["tx_power_dbm"] = tx.tolist()
    return {**opt_results, "prop_params": live_prop}, live_specs


def radius_from_tx(tx_power_dbm: np.ndarray, prop: dict) -> np.ndarray:
    n = max(float(prop["path_loss_exponent"]), 1e-9)
    noise_floor = np.asarray(prop["noise_floor_dbm"], dtype=float)  # 스칼라 또는 배열

    exp = (
        np.asarray(tx_power_dbm, dtype=float)
        - float(prop["path_loss_ref_db"])
        - noise_floor
        - float(prop["sinr_threshold_db"])
    ) / (10.0 * n)

    return np.maximum(1.0, np.power(10.0, exp))



# ---------------------------------------------------------------------------
# Obstacle loading/application
# ---------------------------------------------------------------------------

def load_map_obstacles(
    env: SyntheticEnvironment,
    source: str,
    uploaded_geojson: io.BytesIO | None,
    min_area_m2: float,
    max_obstacles: int | None,
    osm_obstacle_types: list[str] | None = None,
):
    if source == "OSM 지도 데이터":
        if not osm_obstacle_types:
            raise ValueError("OSM 오브젝트 종류를 하나 이상 선택해주세요.")

        try:
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
                obstacle_types=osm_obstacle_types,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'obstacle_types'" not in str(exc):
                raise
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
            )

    elif source == "GeoJSON 업로드":
        if uploaded_geojson is None:
            raise ValueError("GeoJSON 파일을 먼저 업로드해주세요.")
        geo_polygons = geojson_to_polygons(uploaded_geojson.getvalue())
        raw_count = len(geo_polygons)

    else:
        return [], 0

    local_polygons = []

    for polygon in geo_polygons:
        local_polygons.extend(env.geo_to_local_polygons(polygon))

    return filter_polygons(local_polygons, min_area_m2, max_obstacles, coordinates_are_meters=True), raw_count


def apply_obstacle_source(
    env: SyntheticEnvironment,
    source: str,
    uploaded_geojson: io.BytesIO | None,
    min_area_m2: float,
    max_obstacles: int | None,
    obstacle_pattern: str,
    num_obstacles: int,
    osm_obstacle_types: list[str] | None = None,
    append: bool = False,
) -> tuple[int, int]:
    if source == "합성":
        if append:
            before = len(env.obstacles)
            generated = SyntheticEnvironment(
                center_lat=env.center_lat,
                center_lon=env.center_lon,
                width_km=env.width_km,
                height_km=env.height_km,
                resolution_m=env.resolution_m,
            )
            generated.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
            env.append_obstacles(generated.obstacles)
            return len(env.obstacles) - before, num_obstacles

        env.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
        env.remask_traffic()
        return len(env.obstacles), num_obstacles

    polygons, raw_count = load_map_obstacles(
        env,
        source,
        uploaded_geojson,
        min_area_m2,
        max_obstacles,
        osm_obstacle_types=osm_obstacle_types,
    )

    if append:
        env.append_obstacles(polygons)
    else:
        env.replace_obstacles(polygons)

    return len(polygons), raw_count


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def env_dataframe_for_current_frame(env: SyntheticEnvironment) -> pd.DataFrame:
    raw_series = env.get_raw_traffic_series()

    if raw_series is not None:
        flat_traffic = raw_series[env.dynamic_frame_index].ravel()
    else:
        flat_traffic = env.get_raw_traffic_map().ravel()

    obstacle_mask = env.get_obstacle_mask().ravel()

    return pd.DataFrame(
        {
            "lat": env.lat_grid.ravel(),
            "lon": env.lon_grid.ravel(),
            "traffic": flat_traffic,
            "is_obstacle": obstacle_mask,
        }
    )


def compute_status_overlay(
    env: SyntheticEnvironment,
    df: pd.DataFrame,
    opt_results: dict[str, Any] | None,
    opt_stats: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _empty = (np.zeros(len(df), dtype=int), np.zeros(0, dtype=float), np.full(len(df), np.nan))
    if not opt_results or not opt_stats:
        return _empty

    stations = opt_results.get("stations_geo")

    if stations is None or len(stations) == 0:
        return _empty

    station_df = pd.DataFrame(stations)

    if station_df.empty or not {"lat", "lon"}.issubset(station_df.columns):
        return _empty

    station_points = station_df[["lat", "lon"]].values
    station_count = len(station_points)

    prop = opt_results.get("prop_params", {})
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx = coerce_station_tx_power_array(station_specs, station_count, fallback_tx)

    fallback_bw = float(prop.get("bandwidth_mhz", 10.0))
    bw = coerce_station_bandwidth_array(station_specs, station_count, fallback_bw)
    noise_floor_per_station = -174.0 + 10.0 * np.log10(np.maximum(bw, 0.001) * 1e6) + 7.0

    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if len(active_mask) != station_count:
            active_mask = None

    if active_mask is None:
        active_indices = np.arange(station_count)
    else:
        active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return np.zeros(len(df), dtype=int), np.zeros(station_count, dtype=float), np.full(len(df), np.nan)

    active_station_points = station_points[active_indices]
    active_tx = tx[active_indices]
    active_noise_floor = noise_floor_per_station[active_indices]

    traffic_mask = df["traffic"] > 0.1
    grid_points = df.loc[traffic_mask, ["lat", "lon", "traffic"]].values
    grid_indices = np.where(traffic_mask.to_numpy())[0]

    if len(grid_points) == 0:
        return np.zeros(len(df), dtype=int), np.zeros(station_count, dtype=float), np.full(len(df), np.nan)

    x_scale = env.width_m / max(env.lon_max - env.lon_min, 1e-12)
    y_scale = env.height_m / max(env.lat_max - env.lat_min, 1e-12)

    st_x = (active_station_points[:, 1] - env.lon_min) * x_scale
    st_y = (active_station_points[:, 0] - env.lat_min) * y_scale
    st_local = np.column_stack((st_x, st_y))

    gd_x = (grid_points[:, 1] - env.lon_min) * x_scale
    gd_y = (grid_points[:, 0] - env.lat_min) * y_scale
    gd_local = np.column_stack((gd_x, gd_y))

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": active_noise_floor,
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": fallback_bw,
    }

    problem = ProblemInput(
        X=gd_local,
        weights=grid_points[:, 2],
        width_m=env.width_m,
        height_m=env.height_m,
        radius_m=radius_from_tx(active_tx, prop_for_radius),
        capacity=np.full(len(active_indices), 1e10),
        lat_min=env.lat_min,
        lat_max=env.lat_max,
        lon_min=env.lon_min,
        lon_max=env.lon_max,
        path_loss_exponent=prop_for_radius["path_loss_exponent"],
        path_loss_ref_db=prop_for_radius["path_loss_ref_db"],
        tx_power_dbm=active_tx,
        noise_floor_dbm=active_noise_floor,
        sinr_threshold_db=prop_for_radius["sinr_threshold_db"],
        bandwidth_mhz=fallback_bw,
        interference_threshold_dbm=float(prop.get("noise_floor_dbm", -97.0)),
        max_coord_stations=int(prop.get("max_coord_stations", 1)),
    )

    is_cov, srv_idx, best_sinr_db = sinr_coverage(st_local, problem)

    grid_status = np.zeros(len(df), dtype=int)
    overlay_loads = np.zeros(station_count, dtype=float)
    sinr_per_cell = np.full(len(df), np.nan)

    for i in range(len(grid_points)):
        if is_cov[i]:
            grid_status[grid_indices[i]] = 1
            overlay_loads[int(active_indices[int(srv_idx[i])])] += float(grid_points[i, 2])
        sinr_per_cell[grid_indices[i]] = float(best_sinr_db[i])

    return grid_status, overlay_loads, sinr_per_cell


def _traffic_map_for_metrics(env: SyntheticEnvironment, frame_index: int | None = None) -> np.ndarray:
    raw_series = env.get_raw_traffic_series()
    if raw_series is not None:
        max_frame = int(raw_series.shape[0] - 1)
        idx = max(0, min(safe_int(frame_index, env.dynamic_frame_index), max_frame))
        traffic = np.array(raw_series[idx], copy=True, dtype=float)
    else:
        traffic = np.array(env.get_raw_traffic_map(), copy=True, dtype=float)

    obstacle_mask = env.get_obstacle_mask()
    if obstacle_mask.shape == traffic.shape:
        traffic[obstacle_mask] = 0.0

    profile = TIME_PROFILES.get(getattr(env, "time_profile", "flat"), TIME_PROFILES["flat"])
    hour = max(0, min(23, int(getattr(env, "time_hour", 12))))
    time_scale = float(profile[hour])
    if time_scale != 1.0:
        traffic *= time_scale

    return traffic


def _stations_geo_to_local(env: SyntheticEnvironment, stations_geo: Any) -> np.ndarray | None:
    station_df = pd.DataFrame(stations_geo or [])
    if station_df.empty or not {"lat", "lon"}.issubset(station_df.columns):
        return None

    x_scale = env.width_m / max(env.lon_max - env.lon_min, 1e-12)
    y_scale = env.height_m / max(env.lat_max - env.lat_min, 1e-12)

    st_x = (station_df["lon"].to_numpy(dtype=float) - env.lon_min) * x_scale
    st_y = (station_df["lat"].to_numpy(dtype=float) - env.lat_min) * y_scale
    return np.column_stack((st_x, st_y))


def compute_frame_metrics(
    env: SyntheticEnvironment | None,
    opt_results: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
    frame_index: int | None = None,
    active_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if env is None or not opt_results:
        return None

    stations_local = _stations_geo_to_local(env, opt_results.get("stations_geo"))
    if stations_local is None or len(stations_local) == 0:
        return None

    prop = opt_results.get("prop_params", {})
    original_k = len(stations_local)
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx_all = coerce_station_tx_power_array(station_specs, original_k, fallback_tx)
    fallback_bw = float(prop.get("bandwidth_mhz", 10.0))
    bw_all = coerce_station_bandwidth_array(station_specs, original_k, fallback_bw)
    noise_floor_all = -174.0 + 10.0 * np.log10(np.maximum(bw_all, 0.001) * 1e6) + 7.0

    active_mask_applied = False
    active_indices = np.arange(original_k)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if len(mask) == original_k:
            active_mask_applied = True
            active_indices = np.where(mask)[0]

    stations_for_metrics = stations_local[active_indices]
    tx = tx_all[active_indices]
    noise_floor_per_station = noise_floor_all[active_indices]
    k = len(stations_for_metrics)

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": noise_floor_per_station,
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": fallback_bw,
    }

    traffic = _traffic_map_for_metrics(env, frame_index)
    mask = traffic.ravel() > 0
    x_vals = env.x_grid.ravel()[mask]
    y_vals = env.y_grid.ravel()[mask]
    weight_scale = float(opt_results.get("weight_scale", 1.0))
    weights = traffic.ravel()[mask] * weight_scale

    problem = ProblemInput(
        X=np.column_stack((x_vals, y_vals)) if len(weights) > 0 else np.empty((0, 2)),
        weights=weights,
        width_m=env.width_m,
        height_m=env.height_m,
        radius_m=radius_from_tx(tx, prop_for_radius),
        capacity=np.full(k, 1e10),
        lat_min=env.lat_min,
        lat_max=env.lat_max,
        lon_min=env.lon_min,
        lon_max=env.lon_max,
        path_loss_exponent=prop_for_radius["path_loss_exponent"],
        path_loss_ref_db=prop_for_radius["path_loss_ref_db"],
        tx_power_dbm=tx,
        noise_floor_dbm=noise_floor_per_station,
        sinr_threshold_db=prop_for_radius["sinr_threshold_db"],
        bandwidth_mhz=float(prop.get("bandwidth_mhz", fallback_bw)),
        score_mode=opt_results.get("score_mode", "traffic"),
        spectral_efficiency_mode=opt_results.get("spectral_efficiency_mode", "shannon"),
        weight_scale=weight_scale,
        interference_threshold_dbm=float(prop.get("noise_floor_dbm", -97.0)),
        max_coord_stations=int(prop.get("max_coord_stations", 1)),
    )

    metrics = dict(compute_metrics(stations_for_metrics, problem))
    metrics["n_stations"] = k
    metrics["total_station_count"] = original_k
    if active_mask_applied:
        metrics["active_station_count"] = k
        metrics["operation_active_mask_applied"] = True
    metrics["total_tx_power_w"] = float(np.sum(10 ** ((tx - 30) / 10)))
    return metrics


def compute_dynamic_scenario_summary(
    env: SyntheticEnvironment | None,
    opt_results: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
    operation_results: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    series = env.get_raw_traffic_series() if env is not None else None
    if series is None or getattr(series, "ndim", 0) != 3 or series.shape[0] <= 1:
        return None

    traffic_coverage_pct = []
    max_station_load = 0.0
    station_count = len((opt_results or {}).get("stations_geo") or [])
    for frame_idx in range(int(series.shape[0])):
        active_mask = operation_active_mask_for_frame(operation_results, station_count, frame_idx)
        metrics = compute_frame_metrics(
            env,
            opt_results,
            station_specs,
            frame_index=frame_idx,
            active_mask=active_mask,
        )
        if not metrics:
            continue
        total = float(metrics.get("total_traffic", 0.0))
        covered = float(metrics.get("covered_traffic", 0.0))
        traffic_coverage_pct.append((covered / total) * 100.0 if total > 0 else 0.0)
        loads = np.asarray(metrics.get("station_loads", []), dtype=float)
        if loads.size:
            max_station_load = max(max_station_load, float(np.max(loads)))

    if not traffic_coverage_pct:
        return None

    current = max(0, min(int(getattr(env, "dynamic_frame_index", 0)), int(series.shape[0] - 1)))
    return {
        "current_frame": current,
        "max_frame": int(series.shape[0] - 1),
        "avg_traffic_coverage_pct": float(np.mean(traffic_coverage_pct)),
        "worst_traffic_coverage_pct": float(np.min(traffic_coverage_pct)),
        "max_station_load": max_station_load,
    }


def _operation_frame_context(
    env: SyntheticEnvironment,
    opt_results: dict[str, Any],
    station_specs: list[dict[str, Any]] | None,
    frame_index: int,
) -> dict[str, Any] | None:
    stations_local = _stations_geo_to_local(env, opt_results.get("stations_geo"))
    if stations_local is None or len(stations_local) == 0:
        return None

    prop = opt_results.get("prop_params", {})
    k = len(stations_local)
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx = coerce_station_tx_power_array(station_specs, k, fallback_tx)
    fallback_bw = float(prop.get("bandwidth_mhz", 10.0))
    bw = coerce_station_bandwidth_array(station_specs, k, fallback_bw)
    noise_floor_per_station = -174.0 + 10.0 * np.log10(np.maximum(bw, 0.001) * 1e6) + 7.0
    fallback_capacity = capacity_from_bandwidth(fallback_bw)
    capacities = coerce_station_capacity_array(station_specs, k, fallback_capacity)
    capacities = np.maximum(capacities, 1e-9)

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": noise_floor_per_station,
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": fallback_bw,
    }

    traffic = _traffic_map_for_metrics(env, frame_index)
    traffic_mask = traffic.ravel() > 0
    weights = traffic.ravel()[traffic_mask] * float(opt_results.get("weight_scale", 1.0))
    x_vals = env.x_grid.ravel()[traffic_mask]
    y_vals = env.y_grid.ravel()[traffic_mask]
    problem = ProblemInput(
        X=np.column_stack((x_vals, y_vals)) if len(weights) > 0 else np.empty((0, 2)),
        weights=weights,
        width_m=env.width_m,
        height_m=env.height_m,
        radius_m=radius_from_tx(tx, prop_for_radius),
        capacity=capacities,
        lat_min=env.lat_min,
        lat_max=env.lat_max,
        lon_min=env.lon_min,
        lon_max=env.lon_max,
        path_loss_exponent=prop_for_radius["path_loss_exponent"],
        path_loss_ref_db=prop_for_radius["path_loss_ref_db"],
        tx_power_dbm=tx,
        noise_floor_dbm=noise_floor_per_station,
        sinr_threshold_db=prop_for_radius["sinr_threshold_db"],
        bandwidth_mhz=float(prop.get("bandwidth_mhz", fallback_bw)),
        score_mode=opt_results.get("score_mode", "traffic"),
        spectral_efficiency_mode=opt_results.get("spectral_efficiency_mode", "shannon"),
        weight_scale=float(opt_results.get("weight_scale", 1.0)),
        interference_threshold_dbm=float(prop.get("noise_floor_dbm", -97.0)),
        max_coord_stations=int(prop.get("max_coord_stations", 1)),
    )

    return {
        "k": k,
        "problem": problem,
        "stations_local": stations_local,
        "radius_m": np.asarray(problem.radius_m, dtype=float),
        "tx_power_dbm": tx,
        "noise_floor_dbm": noise_floor_per_station,
        "weights": weights,
        "capacities": capacities,
        "sinr_threshold_linear": 10.0 ** (problem.sinr_threshold_db / 10.0),
        "total_traffic": float(np.sum(weights)),
    }


def _operation_loads_for_mask(context: dict[str, Any], active_mask: np.ndarray) -> dict[str, Any]:
    k = int(context["k"])
    weights = np.asarray(context["weights"], dtype=float)
    total_traffic = float(context["total_traffic"])
    loads = np.zeros(k, dtype=float)

    if k == 0 or weights.size == 0:
        return {"loads": loads, "covered_traffic": 0.0, "uncovered_traffic": total_traffic}

    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return {"loads": loads, "covered_traffic": 0.0, "uncovered_traffic": total_traffic}

    active_problem = replace(
        context["problem"],
        radius_m=np.asarray(context["radius_m"], dtype=float)[active_indices],
        capacity=np.asarray(context["capacities"], dtype=float)[active_indices],
        tx_power_dbm=np.asarray(context["tx_power_dbm"], dtype=float)[active_indices],
        noise_floor_dbm=np.asarray(context["noise_floor_dbm"], dtype=float)[active_indices],
    )
    active_sinr = compute_sinr(np.asarray(context["stations_local"], dtype=float)[active_indices], active_problem)
    serving_local = np.argmax(active_sinr, axis=1)
    best_sinr = active_sinr[np.arange(weights.size), serving_local]
    covered_mask = best_sinr >= float(context["sinr_threshold_linear"])
    real_serving = active_indices[serving_local]

    if np.any(covered_mask):
        np.add.at(loads, real_serving[covered_mask], weights[covered_mask])

    covered_traffic = float(np.sum(weights[covered_mask]))
    return {
        "loads": loads,
        "covered_traffic": covered_traffic,
        "uncovered_traffic": max(0.0, total_traffic - covered_traffic),
    }


def _operation_step_cost(
    context: dict[str, Any],
    active_mask: np.ndarray,
    prev_active_mask: np.ndarray,
    params: dict[str, float],
) -> dict[str, Any]:
    active_mask = np.asarray(active_mask, dtype=bool)
    prev_active_mask = np.asarray(prev_active_mask, dtype=bool)
    loads_info = _operation_loads_for_mask(context, active_mask)
    loads = loads_info["loads"]
    capacities = np.asarray(context["capacities"], dtype=float)
    tx_power_dbm = np.asarray(context["tx_power_dbm"], dtype=float)
    tx_power_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    active_power_w = tx_power_w * OPERATION_ACTIVE_POWER_TX_MULTIPLIER
    sleep_power_w = tx_power_w * OPERATION_SLEEP_POWER_TX_MULTIPLIER

    load_ratio = np.divide(loads, capacities, out=np.zeros_like(loads), where=capacities > 0)
    load_ratio = np.clip(load_ratio, 0.0, 1.0)
    energy_cost = float(
        np.sum(np.where(
            active_mask,
            active_power_w + load_ratio * params["load_power_w"],
            sleep_power_w,
        ))
    )
    switching_cost = float(np.sum(active_mask & (~prev_active_mask)) * params["switching_cost"])
    overload_traffic = float(np.sum(np.maximum(0.0, loads - capacities)))
    penalty_cost = float(
        loads_info["uncovered_traffic"] * params["uncovered_penalty"]
        + overload_traffic * params["overload_penalty"]
    )
    step_opex = energy_cost + switching_cost + penalty_cost

    return {
        "loads": loads,
        "active_count": int(np.sum(active_mask)),
        "energy_cost": energy_cost,
        "switching_cost": switching_cost,
        "penalty_cost": penalty_cost,
        "uncovered_traffic": float(loads_info["uncovered_traffic"]),
        "overload_traffic": overload_traffic,
        "covered_traffic": float(loads_info["covered_traffic"]),
        "step_opex": float(step_opex),
        "active_mask": active_mask.tolist(),
    }


def _summarize_operation_history(
    policy: str,
    station_count: int,
    history: list[dict[str, Any]],
    policy_note: str | None = None,
) -> dict[str, Any]:
    total_energy = float(sum(row["energy_cost"] for row in history))
    total_switching = float(sum(row["switching_cost"] for row in history))
    total_penalty = float(sum(row["penalty_cost"] for row in history))
    active_counts = [int(row["active_count"]) for row in history]

    return {
        "policy": policy,
        "policy_note": policy_note,
        "frame_count": len(history),
        "station_count": int(station_count),
        "total_opex": total_energy + total_switching + total_penalty,
        "total_energy_cost": total_energy,
        "total_switching_cost": total_switching,
        "total_penalty_cost": total_penalty,
        "total_uncovered_traffic": float(sum(row["uncovered_traffic"] for row in history)),
        "total_overload_traffic": float(sum(row["overload_traffic"] for row in history)),
        "avg_active_count": float(np.mean(active_counts)) if active_counts else 0.0,
        "min_active_count": int(np.min(active_counts)) if active_counts else 0,
        "max_active_count": int(np.max(active_counts)) if active_counts else 0,
        "history": history,
    }


def evaluate_operation_optimization(
    env: SyntheticEnvironment | None,
    opt_results: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
    policy: str,
    operation_params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    series = env.get_raw_traffic_series() if env is not None else None
    if env is None or not opt_results or series is None or getattr(series, "ndim", 0) != 3 or series.shape[0] <= 1:
        return None

    policy = normalize_operation_policy(policy)
    params = normalize_operation_params(operation_params)
    frame_count = int(series.shape[0])
    first_context = _operation_frame_context(env, opt_results, station_specs, 0)
    if not first_context:
        return None

    k = int(first_context["k"])
    prev_active_mask = np.ones(k, dtype=bool)
    dqn_agent = None
    dqn_note = None
    history: list[dict[str, Any]] = []
    baseline_history: list[dict[str, Any]] = []

    if policy == "dqn":
        try:
            from optimizers.drl.dqn_agent import IndependentDQNAgent
            dqn_agent = IndependentDQNAgent(
                state_dim=2,
                action_dim=2,
                lr=params["dqn_lr"],
                gamma=params["dqn_gamma"],
            )
            dqn_agent.epsilon = params["dqn_epsilon"]
            dqn_agent.epsilon_decay = params["dqn_epsilon_decay"]
            dqn_agent.epsilon_min = params["dqn_epsilon_min"]
        except Exception as exc:  # pragma: no cover - optional torch path
            dqn_note = f"DQN 초기화 실패로 threshold 정책을 사용했습니다: {exc}"
            policy = "threshold"

    for frame_idx in range(frame_count):
        context = first_context if frame_idx == 0 else _operation_frame_context(env, opt_results, station_specs, frame_idx)
        if not context:
            continue

        all_active = np.ones(k, dtype=bool)
        baseline = _operation_loads_for_mask(context, all_active)
        baseline_loads = baseline["loads"]
        baseline_step = _operation_step_cost(context, all_active, np.ones(k, dtype=bool), params)
        baseline_step["time_step"] = frame_idx
        baseline_history.append({name: value for name, value in baseline_step.items() if name != "loads"})
        capacities = np.asarray(context["capacities"], dtype=float)
        threshold = float(params["sleep_threshold_mbps"])

        if policy == "always-on":
            active_mask = all_active
        elif policy == "threshold":
            active_mask = baseline_loads > threshold
            if k > 0 and not np.any(active_mask):
                active_mask[int(np.argmax(baseline_loads))] = True
        elif policy == "two-threshold":
            active_mask = prev_active_mask.copy()
            active_mask[baseline_loads > threshold * params["wake_threshold_multiplier"]] = True
            active_mask[baseline_loads < threshold] = False
            if k > 0 and not np.any(active_mask):
                active_mask[int(np.argmax(baseline_loads))] = True
        elif policy == "greedy-off":
            active_mask = all_active.copy()
            best_cost = _operation_step_cost(context, active_mask, prev_active_mask, params)["step_opex"]
            improved = True
            while improved and np.sum(active_mask) > 1:
                improved = False
                best_candidate = active_mask
                for station_idx in np.where(active_mask)[0]:
                    candidate = active_mask.copy()
                    candidate[station_idx] = False
                    candidate_cost = _operation_step_cost(context, candidate, prev_active_mask, params)["step_opex"]
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_candidate = candidate
                        improved = True
                active_mask = best_candidate
        elif policy == "dqn" and dqn_agent is not None:
            states = np.column_stack((
                np.clip(baseline_loads / capacities, 0.0, 1.0),
                prev_active_mask.astype(float),
            ))
            actions = dqn_agent.get_actions_batch(states, train=True)
            active_mask = actions.astype(bool)
            if k > 0 and not np.any(active_mask):
                active_mask[int(np.argmax(baseline_loads))] = True
        else:
            active_mask = all_active

        step = _operation_step_cost(context, active_mask, prev_active_mask, params)
        step["time_step"] = frame_idx
        history.append({name: value for name, value in step.items() if name != "loads"})

        if policy == "dqn" and dqn_agent is not None:
            rewards = np.full(k, -float(step["step_opex"]) / 100.0, dtype=float)
            next_states = np.column_stack((
                np.clip(step["loads"] / capacities, 0.0, 1.0),
                active_mask.astype(float),
            ))
            dqn_agent.train_step(states, active_mask.astype(int), rewards, next_states)
            dqn_agent.update_epsilon()

        prev_active_mask = active_mask.copy()

    if not history:
        return None

    result = _summarize_operation_history(policy, k, history, dqn_note)
    result["baseline"] = _summarize_operation_history("always-on", k, baseline_history)
    result["operation_params"] = {
        name: params[name]
        for name in operation_param_names_for_policy(policy)
        if name in params
    }
    return result


def operation_summary_rows(result: dict[str, Any] | None) -> list[dict[str, str]]:
    if not result:
        return []

    return [
        {"항목": "운영 정책", "값": str(result.get("policy", "-"))},
        {"항목": "프레임 수", "값": f"{int(result.get('frame_count', 0))}"},
        {"항목": "총 OPEX", "값": f"{float(result.get('total_opex', 0.0)):.1f}"},
        {"항목": "Energy cost", "값": f"{float(result.get('total_energy_cost', 0.0)):.1f}"},
        {"항목": "Switching cost", "값": f"{float(result.get('total_switching_cost', 0.0)):.1f}"},
        {"항목": "Penalty cost", "값": f"{float(result.get('total_penalty_cost', 0.0)):.1f}"},
        {"항목": "평균 Active 기지국", "값": f"{float(result.get('avg_active_count', 0.0)):.1f}"},
        {"항목": "Active 범위", "값": f"{result.get('min_active_count', '-')} - {result.get('max_active_count', '-')}"},
        {"항목": "미커버 트래픽", "값": format_metric_value("트래픽", result.get("total_uncovered_traffic"))},
        {"항목": "용량 초과 트래픽", "값": format_metric_value("트래픽", result.get("total_overload_traffic"))},
    ]


def operation_comparison_rows(result: dict[str, Any] | None) -> list[dict[str, str]]:
    if not result or not result.get("baseline"):
        return []

    before = result["baseline"]
    after = result
    before_label = f"전({before.get('policy', 'always-on')})"
    after_label = f"후({after.get('policy', '-')})"

    def numeric(key: str, source: dict[str, Any]) -> float:
        try:
            return float(source.get(key, 0.0))
        except Exception:
            return 0.0

    def delta_text(key: str) -> str:
        before_value = numeric(key, before)
        after_value = numeric(key, after)
        delta = after_value - before_value
        pct = (delta / before_value * 100.0) if abs(before_value) > 1e-9 else None
        if pct is None:
            return f"{delta:+.1f}"
        return f"{delta:+.1f} ({pct:+.1f}%)"

    rows = [
        ("총 OPEX", "total_opex", lambda v: f"{v:.1f}"),
        ("Energy cost", "total_energy_cost", lambda v: f"{v:.1f}"),
        ("Switching cost", "total_switching_cost", lambda v: f"{v:.1f}"),
        ("Penalty cost", "total_penalty_cost", lambda v: f"{v:.1f}"),
        ("평균 Active 기지국", "avg_active_count", lambda v: f"{v:.1f}"),
        ("미커버 트래픽", "total_uncovered_traffic", lambda v: format_metric_value("트래픽", v)),
        ("용량 초과 트래픽", "total_overload_traffic", lambda v: format_metric_value("트래픽", v)),
    ]

    return [
        {
            "항목": label,
            before_label: formatter(numeric(key, before)),
            after_label: formatter(numeric(key, after)),
            "변화": delta_text(key),
        }
        for label, key, formatter in rows
    ]


def operation_history_figure(result: dict[str, Any] | None) -> go.Figure:
    fig = go.Figure()
    history = (result or {}).get("history") or []
    if not history:
        fig.add_annotation(
            text="운영 최적화 결과가 없습니다.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
    else:
        steps = [row["time_step"] for row in history]
        fig.add_trace(go.Bar(
            x=steps,
            y=[row["energy_cost"] for row in history],
            name="Energy",
            marker_color="#7132f5",
        ))
        fig.add_trace(go.Bar(
            x=steps,
            y=[row["switching_cost"] for row in history],
            name="Switching",
            marker_color="#9ca3af",
        ))
        fig.add_trace(go.Bar(
            x=steps,
            y=[row["penalty_cost"] for row in history],
            name="Penalty",
            marker_color="#cf202f",
        ))
        fig.add_trace(go.Scatter(
            x=steps,
            y=[row["active_count"] for row in history],
            mode="lines+markers",
            name="Active BS",
            yaxis="y2",
            line={"color": "#0a0b0d", "width": 2},
        ))

    fig.update_layout(
        barmode="stack",
        height=260,
        margin={"l": 45, "r": 45, "t": 16, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Frame", "gridcolor": "#dedee5"},
        yaxis={"title": "Cost", "gridcolor": "#dedee5"},
        yaxis2={"title": "Active BS", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def render_operation_status(result: dict[str, Any] | None):
    if not result:
        return html.Span("운영 최적화 결과가 없습니다.", style={"color": "#b91c1c"})

    note = result.get("policy_note")
    children = [
        html.Div(
            f"완료 | 정책 {result.get('policy')} | 총 OPEX {float(result.get('total_opex', 0.0)):.1f}",
            style={"color": "#166534", "fontWeight": "600"},
        ),
        html.Div(
            f"Energy {float(result.get('total_energy_cost', 0.0)):.1f} / "
            f"Switching {float(result.get('total_switching_cost', 0.0)):.1f} / "
            f"Penalty {float(result.get('total_penalty_cost', 0.0)):.1f}",
            style={"marginTop": "3px"},
        ),
    ]
    if note:
        children.append(html.Div(note, style={"marginTop": "3px", "color": "#b45309"}))
    return html.Div(children)


def build_traffic_geojson(
    env: SyntheticEnvironment,
    df: pd.DataFrame,
    map_layer_mode: str,
    status_list: np.ndarray,
    sinr_per_cell: np.ndarray | None = None,
) -> dict[str, Any]:
    lat_step = (env.lat_max - env.lat_min) / max(env.rows, 1)
    lon_step = (env.lon_max - env.lon_min) / max(env.cols, 1)

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    traffics = df["traffic"].to_numpy()
    is_obstacles = df["is_obstacle"].to_numpy(dtype=bool)

    # 실제 트래픽 범위로 정규화 (단위가 Mbps든 추상값이든 동일하게 동작)
    max_traffic = float(traffics.max()) if len(traffics) > 0 else 1.0
    if max_traffic <= 0:
        max_traffic = 1.0

    features = []

    for idx in range(len(df)):
        lat = float(lats[idx])
        lon = float(lons[idx])
        traffic = float(traffics[idx])
        is_obstacle = bool(is_obstacles[idx])

        norm = traffic / max_traffic  # 0~1 범위
        status_text = "N/A"

        if is_obstacle:
            # 장애물: 회색조 (밝기 40~70%, 불투명도 고정)
            gray = int(100 + norm * 70)
            color = f"rgb({gray},{gray},{gray})"
            opacity = 0.75
            status_text = "Obstacle"
        elif map_layer_mode == "커버리지 상태 (Status)" and len(status_list) > idx:
            status = int(status_list[idx])
            color = "#0000ff" if status == 1 else "#ff0000"
            opacity = min(norm * 0.7 + 0.2, 0.9)
            status_text = "Covered" if status == 1 else "Uncovered"
        else:
            color = "#ff0000"
            opacity = min(norm * 0.8, 0.8)

        min_lat, max_lat = lat - lat_step / 2, lat + lat_step / 2
        min_lon, max_lon = lon - lon_step / 2, lon + lon_step / 2

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]],
                },
                "properties": {
                    "traffic": round(traffic, 2),
                    "is_obstacle": is_obstacle,
                    "obstacle": "Obstacle" if is_obstacle else "Open",
                    "status": status_text,
                    "sinr_db": (
                        round(float(sinr_per_cell[idx]), 1)
                        if sinr_per_cell is not None and idx < len(sinr_per_cell)
                           and not np.isnan(sinr_per_cell[idx])
                        else None
                    ),
                    "fillColor": color,
                    "fillOpacity": float(opacity),
                    "interactive": True,
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def build_station_popup(
    station_idx: int,
    lat: float,
    lon: float,
    load: float,
    tx_power: float,
    radius_m: float,
    bandwidth: float = 10.0,
    operation_status: str | None = None,
):
    status_suffix = f" | {operation_status}" if operation_status else ""
    return dl.Popup(
        children=[
            html.Div(
                [
                    html.B(f"Station #{station_idx + 1}", style={"fontSize": "14px",
                                                                   "marginBottom": "6px",
                                                                   "display": "block"}),

                    html.Div(f"Lat: {lat:.6f}", style={"color": "#555"}),
                    html.Div(f"Lon: {lon:.6f}", style={"color": "#555"}),

                    html.Hr(style={"margin": "8px 0"}),

                    html.Div(
                        f"Load: {load:.1f}{status_suffix}",
                        id={"type": "station-popup-load", "index": station_idx},
                    ),
                    html.Div(f"Tx Power: {tx_power:.1f} dBm"),
                    html.Div(f"예상 커버 반경: {radius_m:.0f} m (시각화 전용)"),

                    html.Hr(style={"margin": "8px 0"}),

                    html.Label("Tx Power (dBm)", style={"display": "block", "fontWeight": "700",
                                                              "marginTop": "6px"}),
                    dcc.Slider(
                        id={"type": "station-tx-input", "index": station_idx},
                        min=20, max=50, step=1,
                        value=float(tx_power),
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks={20: "20", 30: "30", 43: "43", 50: "50"},
                    ),

                    html.Label("대역폭 (MHz)", style={"display": "block", "fontWeight": "700",
                                                      "marginTop": "6px"}),
                    dcc.Slider(
                        id={"type": "station-bandwidth-input", "index": station_idx},
                        min=1, max=100, step=1,
                        value=float(bandwidth),
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks={1: "1", 10: "10", 20: "20", 50: "50", 100: "100"},
                    ),
                ],
                style={
                    "minWidth": "240px",
                    "fontFamily": "sans-serif",
                    "fontSize": "12px",
                    "lineHeight": "1.45",
                },
            )
        ],
        maxWidth=320,
    )


def _station_layer_data(
    opt_results: dict[str, Any],
    opt_stats: dict[str, Any],
    station_specs: list[dict[str, Any]] | None,
    overlay_loads: np.ndarray,
):
    """공통 계산: stations, tx, bw, radii, loads 반환."""
    stations = pd.DataFrame(opt_results.get("stations_geo", []))
    if stations.empty or not {"lat", "lon"}.issubset(stations.columns):
        return None

    prop = opt_results.get("prop_params", {})
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx = coerce_station_tx_power_array(station_specs, len(stations), fallback_tx)

    fallback_bw = float(prop.get("bandwidth_mhz", 10.0))
    bw = coerce_station_bandwidth_array(station_specs, len(stations), fallback_bw)

    noise_floor_per_station = -174.0 + 10.0 * np.log10(np.maximum(bw, 0.001) * 1e6) + 7.0

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": noise_floor_per_station,
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": fallback_bw,
    }
    radii = radius_from_tx(tx, prop_for_radius)

    return {
        "stations": stations,
        "tx": tx, "bw": bw, "radii": radii,
        "fallback_tx": fallback_tx, "fallback_bw": fallback_bw,
        "overlay_loads": overlay_loads,
    }


def build_station_circles(
    opt_results: dict[str, Any],
    opt_stats: dict[str, Any],
    station_specs: list[dict[str, Any]] | None,
    selected_station_idx: int | None,
    overlay_loads: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> list[Any]:
    """커버 반경 원만 반환 (station-specs 변경 시 실시간 갱신, 팝업 없음)."""
    d = _station_layer_data(opt_results, opt_stats, station_specs, overlay_loads)
    if d is None:
        return []

    stations = d["stations"]
    st_lats = stations["lat"].to_numpy()
    st_lons = stations["lon"].to_numpy()
    layers = []

    for i in range(len(stations)):
        selected = selected_station_idx == i
        active = True if active_mask is None or i >= len(active_mask) else bool(active_mask[i])
        color = "#facc15" if selected and active else "#f59e0b" if selected else "#149e61" if active else "#6b7280"
        fill_color = "#149e61" if active else "#9ca3af"
        layers.append(
            dl.Circle(
                center=[float(st_lats[i]), float(st_lons[i])],
                radius=float(d["radii"][i]) if i < len(d["radii"]) else 300.0,
                color=color,
                weight=3 if selected else 1,
                dashArray=None if active else "5 5",
                fill=True,
                fillColor=fill_color,
                fillOpacity=0.18 if selected and active else 0.1 if active else 0.03,
                interactive=False,
            )
        )
    return layers


def build_station_markers(
    opt_results: dict[str, Any],
    opt_stats: dict[str, Any],
    station_specs: list[dict[str, Any]] | None,
    selected_station_idx: int | None,
    overlay_loads: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> list[Any]:
    """팝업 포함 마커만 반환 (opt-meta / selected-station 변경 시에만 갱신)."""
    d = _station_layer_data(opt_results, opt_stats, station_specs, overlay_loads)
    if d is None:
        return []

    stations = d["stations"]
    st_lats = stations["lat"].to_numpy()
    st_lons = stations["lon"].to_numpy()
    layers = []

    for i in range(len(stations)):
        active = True if active_mask is None or i >= len(active_mask) else bool(active_mask[i])
        operation_status = "ON" if active else "SLEEP/OFF"
        lat = float(st_lats[i])
        lon = float(st_lons[i])
        load = float(d["overlay_loads"][i]) if active and i < len(d["overlay_loads"]) else 0.0
        radius_m = float(d["radii"][i]) if i < len(d["radii"]) else 300.0
        tx_i = float(d["tx"][i]) if i < len(d["tx"]) else d["fallback_tx"]
        bw_i = float(d["bw"][i]) if i < len(d["bw"]) else d["fallback_bw"]
        selected = selected_station_idx == i

        popup = build_station_popup(
            station_idx=int(i),
            lat=lat, lon=lon,
            load=load,
            tx_power=tx_i, radius_m=radius_m, bandwidth=bw_i,
            operation_status=operation_status if active_mask is not None else None,
        )
        tooltip = dl.Tooltip(
            f"Station #{i + 1} | {operation_status}" + (" (선택됨)" if selected else "")
        )

        # 중요: n_clicks=0을 명시해 dash-leaflet이 클릭 가능한 레이어로 인식하도록 한다.
        if STATION_PIN_MARKER_ENABLED:
            layers.append(dl.Marker(
                id={"type": "station-marker", "index": int(i)},
                position=[lat, lon],
                opacity=1.0 if active else 0.35,
                zIndexOffset=250 if active else -250,
                interactive=True, n_clicks=0, bubblingMouseEvents=False,
                children=[tooltip, popup],
            ))
        else:
            layers.append(dl.CircleMarker(
                id={"type": "station-marker", "index": int(i)},
                center=[lat, lon], radius=13,
                color="#149e61" if active else "#6b7280",
                weight=4 if active else 2,
                fill=True,
                fillColor="#149e61" if active else "#9ca3af",
                fillOpacity=0.95 if active else 0.35,
                interactive=True, n_clicks=0, bubblingMouseEvents=False,
                children=[tooltip, popup],
            ))

    return layers



# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def metric_card(title: str, value: str):
    return html.Div(
        [
            html.Div(title, className="metric-card__title"),
            html.Div(value, className="metric-card__value"),
        ],
        className="metric-card",
    )


def _empty_stats_cards():
    return [
        metric_card("총 트래픽", "-"),
        metric_card("커버된 트래픽", "-"),
        metric_card("커버된 면적", "-"),
        metric_card("평균 SINR", "-"),
        metric_card("총 처리량", "-"),
        metric_card("기지국 수", "-"),
        metric_card("에너지 효율", "-"),
    ]


def analysis_section(title: str, children: list[Any]):
    return html.Section(
        [
            html.H3(title, className="analysis-section__title"),
            *children,
        ],
        className="analysis-section",
    )


def analysis_empty(message: str):
    return html.Div(message, className="analysis-empty")


def compact_table(rows: list[dict[str, Any]], page_size: int = 8):
    if not rows:
        return analysis_empty("표시할 결과가 없습니다.")

    columns = [{"name": key, "id": key} for key in rows[0].keys()]
    return dash_table.DataTable(
        data=rows,
        columns=columns,
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontSize": "12px",
            "padding": "6px 8px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"fontSize": "12px", "fontWeight": "700"},
    )


def format_metric_value(metric_name: str, value: Any) -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return "-"
            lowered = metric_name.lower()
            if "pct" in lowered or "(%)" in metric_name or "커버율" in metric_name:
                return f"{numeric:.1f}%"
            if "sinr" in lowered or "SINR" in metric_name:
                return f"{numeric:.1f} dB"
            if "throughput" in lowered or "처리량" in metric_name:
                return f"{numeric:.1f} Mbps"
            if "traffic" in lowered or "트래픽" in metric_name or "load" in lowered or "부하" in metric_name:
                return f"{numeric:.2f} Mbps" if abs(numeric) < 1e4 else f"{numeric:.0f}"
            if "elapsed" in lowered or "시간" in metric_name:
                return f"{numeric:.2f}s"
            if "score" in lowered:
                return f"{numeric:.2f}"
            if numeric.is_integer():
                return f"{int(numeric)}"
            return f"{numeric:.3g}"
    except Exception:
        pass
    return str(value)


def environment_summary_rows(env: SyntheticEnvironment | None) -> list[dict[str, str]]:
    if env is None:
        return []

    raw_series = env.get_raw_traffic_series()
    dynamic_label = "동적" if raw_series is not None and raw_series.shape[0] > 1 else "정적"
    frame_label = (
        f"{int(getattr(env, 'dynamic_frame_index', 0)) + 1} / {int(raw_series.shape[0])}"
        if raw_series is not None and raw_series.shape[0] > 1
        else "-"
    )

    raw_map = _traffic_map_for_metrics(env)
    obstacle_count = len(getattr(env, "obstacles", []) or [])

    return [
        {"항목": "영역 크기", "값": f"{env.width_km:.2f} km x {env.height_km:.2f} km"},
        {"항목": "해상도", "값": f"{env.resolution_m:.0f} m"},
        {"항목": "격자 수", "값": f"{env.rows} x {env.cols} ({env.rows * env.cols} cells)"},
        {"항목": "트래픽 모드", "값": dynamic_label},
        {"항목": "현재 프레임", "값": frame_label},
        {"항목": "총 트래픽", "값": format_metric_value("트래픽", float(np.sum(raw_map)))},
        {"항목": "오브젝트 수", "값": f"{obstacle_count}"},
    ]


def current_metrics_rows(metrics: dict[str, Any] | None) -> list[dict[str, str]]:
    if not metrics:
        return []

    total_t = float(metrics.get("total_traffic", 0.0))
    covered_t = float(metrics.get("covered_traffic", 0.0))
    total_area = float(metrics.get("total_area", 0.0))
    covered_area = float(metrics.get("covered_area", 0.0))
    total_tp = float(metrics.get("total_throughput_mbps", 0.0))
    total_tx_w = metrics.get("total_tx_power_w")
    traffic_cov = (covered_t / total_t) * 100.0 if total_t > 0 else 0.0
    area_cov = (covered_area / total_area) * 100.0 if total_area > 0 else 0.0
    energy_eff = (total_tp / float(total_tx_w)) if total_tx_w else None

    return [
        {"항목": "총 트래픽", "값": format_metric_value("트래픽", total_t)},
        {"항목": "커버된 트래픽", "값": f"{format_metric_value('트래픽', covered_t)} ({traffic_cov:.1f}%)"},
        {"항목": "커버된 면적", "값": f"{int(covered_area)} cells ({area_cov:.1f}%)"},
        {"항목": "평균 SINR", "값": format_metric_value("SINR", metrics.get("mean_sinr_db"))},
        {"항목": "총 처리량", "값": format_metric_value("처리량", total_tp)},
        {"항목": "기지국 수", "값": format_metric_value("count", metrics.get("n_stations"))},
        {"항목": "에너지 효율", "값": f"{energy_eff:.3f} Mbps/W" if energy_eff is not None else "-"},
    ]


def dynamic_frame_figure(
    env: SyntheticEnvironment | None,
    opt_results: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
    operation_results: dict[str, Any] | None = None,
) -> go.Figure:
    fig = go.Figure()
    series = env.get_raw_traffic_series() if env is not None else None
    if series is None or getattr(series, "ndim", 0) != 3 or series.shape[0] <= 1:
        fig.add_annotation(
            text="동적 트래픽 결과가 없습니다.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )
    else:
        frames = list(range(int(series.shape[0])))
        coverage_pct: list[float] = []
        total_traffic: list[float] = []
        max_loads: list[float] = []
        station_count = len((opt_results or {}).get("stations_geo") or [])
        for frame_idx in frames:
            active_mask = operation_active_mask_for_frame(operation_results, station_count, frame_idx)
            metrics = compute_frame_metrics(
                env,
                opt_results,
                station_specs,
                frame_index=frame_idx,
                active_mask=active_mask,
            )
            if metrics:
                total = float(metrics.get("total_traffic", 0.0))
                covered = float(metrics.get("covered_traffic", 0.0))
                coverage_pct.append((covered / total) * 100.0 if total > 0 else 0.0)
                total_traffic.append(total)
                loads = np.asarray(metrics.get("station_loads", []), dtype=float)
                max_loads.append(float(np.max(loads)) if loads.size else 0.0)
            else:
                frame_traffic = np.asarray(series[frame_idx], dtype=float)
                total_traffic.append(float(np.sum(frame_traffic)))
                coverage_pct.append(0.0)
                max_loads.append(0.0)

        fig.add_trace(go.Scatter(
            x=frames,
            y=coverage_pct,
            mode="lines+markers",
            name="트래픽 커버율(%)",
            line={"color": "#7132f5", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=frames,
            y=max_loads,
            mode="lines+markers",
            name="최대 기지국 부하",
            yaxis="y2",
            line={"color": "#149e61", "width": 2},
        ))
        current = max(0, min(int(getattr(env, "dynamic_frame_index", 0)), int(series.shape[0] - 1)))
        fig.add_vline(x=current, line={"color": "#cf202f", "width": 1, "dash": "dot"})
        fig.update_layout(
            yaxis={"title": "커버율(%)", "gridcolor": "#dedee5"},
            yaxis2={"title": "부하", "overlaying": "y", "side": "right", "showgrid": False},
        )

    fig.update_layout(
        height=260,
        margin={"l": 35, "r": 45, "t": 16, "b": 32},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12},
        xaxis={"title": "Frame", "gridcolor": "#dedee5"},
    )
    return fig


def station_analysis_rows(
    opt_results: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not opt_results:
        return []

    stations = opt_results.get("stations_geo") or []
    prop = opt_results.get("prop_params", {})
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    fallback_bw = float(prop.get("bandwidth_mhz", 10.0))
    loads = np.asarray((metrics or {}).get("station_loads", []), dtype=float)
    tx = coerce_station_tx_power_array(station_specs, len(stations), fallback_tx)
    bw = coerce_station_bandwidth_array(station_specs, len(stations), fallback_bw)

    rows = []
    for i, station in enumerate(stations):
        rows.append({
            "기지국": i + 1,
            "위도": round(safe_float(station.get("lat"), 0.0), 6),
            "경도": round(safe_float(station.get("lon"), 0.0), 6),
            "부하": format_metric_value("부하", loads[i] if i < len(loads) else 0.0),
            "송신전력": f"{tx[i]:.0f} dBm" if i < len(tx) else "-",
            "대역폭": f"{bw[i]:.0f} MHz" if i < len(bw) else "-",
        })
    return rows


def convergence_figure(opt_results: dict[str, Any] | None) -> go.Figure:
    fig = go.Figure()
    history = (opt_results or {}).get("history") or []
    xs: list[int] = []
    ys: list[float] = []

    for i, entry in enumerate(history):
        value = None
        if isinstance(entry, dict):
            value = entry.get("best_score", entry.get("score", entry.get("gen_score")))
            x_val = entry.get("iter", i)
        else:
            value = entry
            x_val = i
        try:
            if value is not None:
                ys.append(float(value))
                xs.append(safe_int(x_val, i))
        except Exception:
            continue

    if not ys and opt_results and opt_results.get("score") is not None:
        xs = [0]
        ys = [float(opt_results["score"])]

    if ys:
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": "#7132f5", "width": 2},
            name="Score",
        ))
    else:
        fig.add_annotation(
            text="최적화 수렴 이력이 없습니다.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
        )

    fig.update_layout(
        height=240,
        margin={"l": 40, "r": 15, "t": 16, "b": 32},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Iteration", "gridcolor": "#dedee5"},
        yaxis={"title": "Score", "gridcolor": "#dedee5"},
        showlegend=False,
    )
    return fig


def sweep_summary_rows(results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not results:
        return []

    sorted_results = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    rows = []
    for rank, result in enumerate(sorted_results[:10], start=1):
        combo = ", ".join(f"{k}={v:.3g}" for k, v in (result.get("param_combo") or {}).items())
        rows.append({
            "순위": rank,
            "파라미터": combo or "-",
            "Score": format_metric_value("score", result.get("score")),
            "커버 트래픽": format_metric_value("트래픽", result.get("covered_traffic")),
            "커버 면적": format_metric_value("area", result.get("covered_area")),
        })
    return rows


def compare_summary_rows(results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not results:
        return []

    sorted_results = sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    rows = []
    for rank, result in enumerate(sorted_results, start=1):
        rows.append({
            "순위": rank,
            "알고리즘": result.get("algo", "-"),
            "Score": format_metric_value("score", result.get("score")),
            "커버율": format_metric_value("coverage_pct", result.get("coverage_pct")),
            "처리량": format_metric_value("처리량", result.get("total_throughput_mbps")),
            "SINR": format_metric_value("SINR", result.get("mean_sinr_db")),
            "시간": format_metric_value("elapsed", result.get("elapsed_sec")),
        })
    return rows


def _section(title: str, children: list):
    items = ([html.H3(title, className="section-header")] if title else [])
    return html.Div(items + children, style={"marginBottom": "20px"})


def sidebar_layout():
    available_algos = [cls.name for cls in REGISTRY]
    default_algo = available_algos[0] if available_algos else ""

    return html.Aside(
        [
            html.Button("◄", id="left-toggle-btn", n_clicks=0,
                        className="sidebar-toggle-btn"),
            html.Div(
              [
                html.H2("시뮬레이터 제어", style={
                    "marginTop": 0,
                    "fontSize": "15px",
                    "fontWeight": "600",
                    "color": "#0a0b0d",
                    "letterSpacing": "-0.2px",
                    "marginBottom": "16px",
                }),

            _section(
                "",
                [
                    html.Label("격자 크기 (m)"),
                    dcc.Input(
                        id="resolution-m",
                        type="number",
                        min=50,
                        max=500,
                        step=10,
                        value=100,
                        style={"width": "100%"},
                    ),

                    html.Div(
                        [
                            html.H3("트래픽 세부 설정", className="section-header",
                                    style={"marginTop": "16px"}),

                            html.Label(
                                [
                                    html.Span("동적 트래픽 모드", className="dynamic-traffic-toggle__text"),
                                    dcc.Checklist(
                                        id="dynamic-traffic",
                                        options=[{"label": "", "value": "on"}],
                                        value=[],
                                        className="dynamic-traffic-toggle__control",
                                        inputClassName="dynamic-traffic-toggle__input",
                                        labelClassName="dynamic-traffic-toggle__label",
                                    ),
                                ],
                                className="dynamic-traffic-toggle",
                            ),

                            html.Label("트래픽 패턴"),
                            dcc.Dropdown(
                                id="traffic-pattern",
                                options=[{"label": x, "value": x} for x in PATTERN_CHOICES],
                                value=PATTERN_CHOICES[0],
                            ),

                            html.Label("총 면적 수요 (Mbps/km²)"),
                            dcc.Slider(
                                id="area-demand-mbps-km2",
                                min=1,
                                max=200,
                                step=1,
                                value=150,
                                marks={1: "1", 50: "50", 100: "100",
                                       150: "150", 200: "200"},
                                tooltip={"placement": "bottom"},
                            ),
                            html.Div(
                                id="area-demand-cell-display",
                                style={"fontSize": "12px", "color": "#6b7280",
                                       "marginTop": "4px"},
                            ),
                            # 하위 호환용 숨김 입력 (콜백 참조 유지)
                            dcc.Input(id="base-intensity", type="hidden", value=10),
                            dcc.Input(id="max-intensity", type="hidden", value=100),

                            html.Div(
                                [
                                    html.Label("핫스팟 개수"),
                                    dcc.Slider(
                                        id="num-hotspots",
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=5,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("핫스팟 확산 반경 (m)"),
                                    dcc.Slider(
                                        id="spread-m",
                                        min=100,
                                        max=1000,
                                        step=50,
                                        value=300,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="multi-hotspot-controls",
                            ),

                            html.Div(
                                [
                                    html.Label("동적 트래픽 유형"),
                                    dcc.Dropdown(
                                        id="dynamic-traffic-type",
                                        options=DYNAMIC_TRAFFIC_TYPE_OPTIONS,
                                        value=DEFAULT_DYNAMIC_TRAFFIC_TYPE,
                                        clearable=False,
                                    ),

                                    html.Label("프레임 수"),
                                    dcc.Slider(
                                        id="dynamic-time-steps",
                                        min=2,
                                        max=48,
                                        step=1,
                                        value=12,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("시간 변화 강도"),
                                    dcc.Slider(
                                        id="dynamic-variation",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.25,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("공간 이동 범위 (m)"),
                                    dcc.Slider(
                                        id="dynamic-drift-m",
                                        min=0,
                                        max=2000,
                                        step=50,
                                        value=300,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="dynamic-traffic-controls",
                            ),
                        ],
                    ),

                    html.Div(
                        [
                            html.H3("오브젝트 세부 설정", className="section-header",
                                    style={"marginTop": "16px"}),

                            html.Label("오브젝트 소스"),
                            dcc.Dropdown(
                                id="obstacle-source",
                                options=[
                                    {"label": x, "value": x} for x in ["합성", "OSM 지도 데이터", "GeoJSON 업로드"]
                                ],
                                value="합성",
                            ),

                            html.Div(
                                [
                                    html.Label("오브젝트 생성 패턴"),
                                    dcc.Dropdown(
                                        id="obstacle-pattern",
                                        options=[
                                            {"label": x, "value": x}
                                            for x in ["random", "mixed", "circle", "strip", "grid"]
                                        ],
                                        value="random",
                                    ),

                                    html.Label("오브젝트 개수"),
                                    dcc.Slider(
                                        id="num-obstacles",
                                        min=0,
                                        max=10,
                                        step=1,
                                        value=3,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="synthetic-obstacle-controls",
                            ),

                            html.Div(
                                [
                                    html.Label("OSM 오브젝트 타입"),
                                    dcc.Checklist(
                                        id="osm-types",
                                        options=[{"label": x, "value": x} for x in OSM_OBSTACLE_TYPE_LABELS],
                                        value=OSM_OBSTACLE_TYPE_LABELS,
                                    ),
                                ],
                                id="osm-obstacle-controls",
                            ),

                            html.Div(
                                [
                                    dcc.Upload(
                                        id="geojson-upload",
                                        children=html.Div(["GeoJSON 파일을 드래그하거나 클릭해서 업로드"]),
                                        style={
                                            "border": "1px dashed #dedee5",
                                            "borderRadius": "8px",
                                            "padding": "12px",
                                            "textAlign": "center",
                                            "fontSize": "13px",
                                        },
                                        multiple=False,
                                    ),

                                    html.Div(
                                        [
                                            html.Label("최소 오브젝트 면적 (m²)"),
                                            dcc.Slider(
                                                id="min-obstacle-area-m2",
                                                min=0,
                                                max=5000,
                                                step=100,
                                                value=100,
                                                tooltip={"placement": "bottom"},
                                            ),

                                            html.Label("최대 오브젝트 개수"),
                                            dcc.Slider(
                                                id="max-map-obstacles",
                                                min=1,
                                                max=500,
                                                step=10,
                                                value=100,
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        id="geojson-filter-controls",
                                    ),
                                ],
                                id="geojson-obstacle-controls",
                            ),
                        ],
                    ),

                    html.Div(
                        id="custom-region-info",
                        style={
                            "display": "none",
                            "fontSize": "12px",
                            "marginTop": "8px",
                            "padding": "6px 8px",
                            "background": "rgba(20, 158, 97, 0.10)",
                            "border": "1px solid #86efac",
                            "borderRadius": "4px",
                            "color": "#166534",
                        },
                    ),
                    html.Button(
                        "영역 지정",
                        id="create-env-btn",
                        n_clicks=0,
                        className="primary-button",
                    ),
                    html.Button(
                        "초기화",
                        id="clear-region-btn",
                        n_clicks=0,
                        style={
                            "display": "none",
                            "width": "100%",
                            "padding": "6px 12px",
                            "marginTop": "4px",
                            "cursor": "pointer",
                            "background": "#cf202f",
                            "color": "white",
                            "border": "0",
                            "borderRadius": "6px",
                            "fontSize": "12px",
                            "fontWeight": "600",
                        },
                    ),
                    html.Div(id="create-status", style={"fontSize": "13px", "marginTop": "8px"}),
                ],
            ),

            # map-layer-mode hidden — always "커버리지 상태 (Status)"
            html.Div(
                dcc.RadioItems(
                    id="map-layer-mode",
                    options=[{"label": "커버리지 상태 (Status)", "value": "커버리지 상태 (Status)"}],
                    value="커버리지 상태 (Status)",
                ),
                style={"display": "none"},
            ),

              ],
              id="left-sidebar-body",
              className="sidebar-body",
              style={"overflowY": "auto", "flex": "1"},
            ),
        ],
        id="left-sidebar",
        style={
            "width": "320px",
            "minWidth": "320px",
            "height": "100vh",
            "display": "flex",
            "flexDirection": "column",
            "background": "#ffffff",
            "borderRight": "1px solid #dedee5",
            "boxSizing": "border-box",
            "padding": "0",
            "transition": "width 0.2s ease, min-width 0.2s ease",
        },
    )


def _mode2_accordion_item(cls) -> html.Div:
    """모드 2 아코디언 아이템: 체크박스 + 알고리즘명 헤더 + 하이퍼파라미터 바디."""
    algo = cls.name
    optimizer = cls()

    # 하이퍼파라미터 컨트롤 (기존 mode2-hp ID 그대로 유지)
    rows = []
    for hp in optimizer.hyperparams or []:
        if hp.kind == "bool":
            ctrl = dcc.Checklist(
                id={"type": "mode2-hp", "algo": algo, "param": hp.name},
                options=[{"label": "", "value": "on"}],
                value=["on"] if hp.default else [],
                style={"display": "inline-block"},
            )
        else:
            ctrl = dcc.Input(
                id={"type": "mode2-hp", "algo": algo, "param": hp.name},
                type="number",
                value=hp.default,
                step=1 if hp.kind == "int" else "any",
                style={"width": "90px", "fontSize": "12px", "padding": "2px 4px"},
            )
        rows.append(html.Div(
            [
                html.Span(hp.name, style={"fontSize": "11px", "color": "#6b7280",
                                          "minWidth": "110px", "display": "inline-block"}),
                ctrl,
            ],
            style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "4px"},
        ))
    if not rows:
        rows = [html.Div("하이퍼파라미터 없음", style={"fontSize": "11px", "color": "#9ca3af"})]

    return html.Div([
        # 헤더: 체크박스 + 알고리즘명 + 토글 버튼
        html.Div([
            dcc.Checklist(
                id={"type": "mode2-algo-check", "algo": algo},
                options=[{"label": "", "value": "on"}],
                value=["on"],
                style={"display": "inline-flex", "alignItems": "center",
                       "marginRight": "6px", "flexShrink": "0"},
            ),
            html.Span(algo, style={"fontSize": "13px", "fontWeight": "600",
                                   "color": "#7132f5", "flex": "1"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                  "padding": "7px 10px", "background": "rgba(133, 91, 251, 0.10)",
                  "borderRadius": "8px", "border": "1px solid rgba(133, 91, 251, 0.40)",
                  "cursor": "default"}),

        # 바디: 하이퍼파라미터
        html.Div(
            rows,
            id={"type": "mode2-body", "algo": algo},
            style={"display": "block", "padding": "8px 10px 6px",
                   "border": "1px solid rgba(133, 91, 251, 0.40)", "borderTop": "none",
                   "borderBottomLeftRadius": "8px", "borderBottomRightRadius": "8px",
                   "background": "#ffffff"},
        ),
    ], style={"marginBottom": "4px"})


def algo_sidebar_layout():
    """알고리즘 설정 패널 — 우측 사이드바에 삽입."""
    available_algos = [cls.name for cls in REGISTRY]
    default_algo = available_algos[0] if available_algos else ""
    return html.Div([
        _section(
            "최적화 목표",
            [
                dcc.Dropdown(
                    id="score-mode",
                    options=[
                        {"label": "트래픽 커버리지", "value": "traffic"},
                        {"label": "커버 셀 수", "value": "cells"},
                    ],
                    value="traffic",
                    clearable=False,
                    style={"marginBottom": "10px"},
                ),
            ],
        ),

        _section(
            "기지국 수",
            [
                dcc.Slider(
                    id="n-stations",
                    min=1,
                    max=100,
                    step=1,
                    value=5,
                    tooltip={"placement": "bottom"},
                ),
            ],
        ),

        _section(
            "전파 모델",
            [
                html.Label("경로 손실 지수 n"),
                dcc.Slider(
                    id="ui-path-loss-exp",
                    min=2.0,
                    max=5.0,
                    step=0.1,
                    value=3.5,
                    tooltip={"placement": "bottom"},
                ),

                html.Label("SINR 임계값 (dB)"),
                dcc.Slider(
                    id="ui-sinr-threshold",
                    min=-10,
                    max=30,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom"},
                ),

                html.Label("CoMP 조율 기지국 수", style={"marginTop": "8px"}),
                dcc.Slider(
                    id="ui-max-coord",
                    min=1,
                    max=19,
                    step=1,
                    value=1,
                    marks={1: "1(없음)", 6: "6", 12: "12", 19: "19"},
                    tooltip={"placement": "bottom"},
                ),

                html.Div(
                    id="noise-caption",
                    style={"fontSize": "12px", "color": "#555", "marginTop": "4px"},
                ),

                html.Div(
                    id="spectral-eff-wrap",
                    children=[
                        html.Label("스펙트럼 효율 모델"),
                        dcc.RadioItems(
                            id="spectral-eff-mode",
                            options=[
                                {"label": "Shannon (이론값)", "value": "shannon"},
                                {"label": "MCS 테이블", "value": "mcs"},
                            ],
                            value="shannon",
                            inline=True,
                            style={"marginTop": "4px"},
                        ),
                    ],
                    style={"display": "none", "marginTop": "8px"},
                ),
            ],
        ),

        _section(
            "알고리즘",
            [
                html.Label("알고리즘 선택"),
                dcc.Dropdown(
                    id="algo-select",
                    options=[{"label": x, "value": x} for x in available_algos],
                    value=default_algo,
                ),
                html.Div(id="hyperparam-controls"),
            ],
        ),

        html.Button(
            "계산 실행",
            id="optimize-btn",
            n_clicks=0,
            className="primary-button",
            style={"marginBottom": "10px"},
        ),

        html.Div(
            _section(
                "운영 최적화",
                [
                    html.Label("운영 정책"),
                    dcc.Dropdown(
                        id="operation-policy",
                        options=OPERATION_POLICY_OPTIONS,
                        value=DEFAULT_OPERATION_POLICY,
                        clearable=False,
                    ),
                    html.Div(id="operation-hyperparam-controls"),
                    html.Button(
                        "운영 최적화 실행",
                        id="operation-run-btn",
                        n_clicks=0,
                        className="primary-button",
                        disabled=True,
                        style={"width": "100%", "marginTop": "10px"},
                    ),
                    html.Div(
                        id="operation-status",
                        style={"fontSize": "12px", "marginTop": "8px", "color": "#6b7280"},
                    ),
                ],
            ),
            id="operation-optimization-section",
            style={"display": "none", "marginTop": "10px"},
        ),

        html.Div(
            dcc.Graph(
                id="sidebar-convergence-chart",
                style={"height": "160px"},
                config={"displayModeBar": False},
            ),
            id="sidebar-convergence-wrap",
            style={"display": "none", "marginTop": "10px"},
        ),

        _section(
            "기지국 모델",
            [
                dcc.RadioItems(
                    id="spec-mode",
                    options=[
                        {"label": "전체 동일", "value": "전체 동일"},
                        {"label": "기지국별 개별", "value": "기지국별 개별"},
                    ],
                    value="전체 동일",
                    inline=True,
                ),

                html.Div(
                    [
                        html.Label("송신 전력 (dBm)"),
                        dcc.Slider(
                            id="ui-tx-power",
                            min=20, max=50, step=1, value=43,
                            tooltip={"placement": "bottom"},
                        ),
                        html.Label("대역폭 (MHz)"),
                        dcc.Slider(
                            id="ui-bandwidth-mhz",
                            min=1, max=100, step=1, value=10,
                            tooltip={"placement": "bottom"},
                        ),
                    ],
                    id="common-spec-wrap",
                    style={"marginTop": "8px"},
                ),

                html.Div(
                    html.Div(id="spec-sliders-container"),
                    id="spec-sliders-wrap",
                    style={"marginTop": "8px", "display": "none"},
                ),
            ],
        ),

        html.Div(
            _section(
                "데이터 내보내기",
                [
                    html.Button("GIS CSV", id="download-gis-btn", n_clicks=0),
                    html.Button(
                        "Local CSV",
                        id="download-local-btn",
                        n_clicks=0,
                        style={"marginLeft": "6px"},
                    ),
                ],
            ),
            style={"display": "none"},
        ),
    ])


def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div(
        [
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="env-meta"),
            dcc.Store(id="opt-meta"),
            dcc.Store(id="operation-meta"),
            dcc.Store(id="range-meta"),
            dcc.Store(id="selected-station", data=None),
            dcc.Store(id="drawn-region-store", data=None),
            dcc.Store(id="custom-region-store", data=None),
            dcc.Store(id="editcontrol-clear-count", data=0),
            dcc.Store(id="algo-history-store", data=None),
            dcc.Store(id="opt-live-store", data=None),
            dcc.Interval(id="opt-poll-interval", interval=750, disabled=True, n_intervals=0),
            dcc.Store(id="station-specs-store", data=[]),
            dcc.Store(id="sweep-meta"),
            dcc.Interval(id="sweep-poll-interval", interval=500, disabled=True, n_intervals=0),
            dcc.Store(id="algo-compare-meta"),
            dcc.Interval(id="algo-compare-poll-interval", interval=1200, disabled=True, n_intervals=0),
            dcc.Store(id="left-sidebar-open", data=True),
            dcc.Store(id="right-sidebar-open", data=True),
            dcc.Store(id="sidebar-resize-dummy"),
            dcc.Store(id="region-draw-activate-dummy", data=False),

            dcc.Download(id="download-gis-csv"),
            dcc.Download(id="download-local-csv"),

            html.Div(
                [
                    sidebar_layout(),

                    html.Div(id="left-resize-handle", className="resize-handle-v"),

                    html.Main(
                        [
                            html.H1(
                                "기지국 위치 최적화 시뮬레이터",
                                style={
                                    "marginTop": 0,
                                    "fontSize": "22px",
                                    "fontWeight": "600",
                                    "color": "#0a0b0d",
                                    "letterSpacing": "-0.3px",
                                    "marginBottom": "16px",
                                },
                            ),

                            html.Div(
                                dcc.RadioItems(
                                    id="main-view-mode",
                                    options=[
                                        {"label": "1", "value": "map"},
                                        {"label": "2", "value": "analysis"},
                                    ],
                                    value="map",
                                    inline=True,
                                    className="view-switch",
                                    inputClassName="view-switch__input",
                                    labelClassName="view-switch__label",
                                ),
                                className="main-view-switch-row",
                            ),

                            html.Div(
                                [
                                    *_empty_stats_cards(),
                                ],
                                id="stats-panel",
                                style={
                                    "display": "flex",
                                    "gap": "10px",
                                    "flexWrap": "wrap",
                                    "marginBottom": "16px",
                                },
                            ),

                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dl.Map(
                                                        id="sim-map",
                                                        center=DEFAULT_CENTER,
                                                        zoom=DEFAULT_ZOOM,
                                                        bounds=None,
                                                        children=[
                                                            dl.TileLayer(
                                                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                                                                attribution="&copy; OpenStreetMap contributors &copy; CARTO",
                                                            ),
                                                            dl.LayerGroup(id="overlay-layers", children=[]),
                                                            dl.LayerGroup(id="station-layer", children=[]),
                                                            dl.FeatureGroup(
                                                                id="region-draw-feature-group",
                                                                children=[
                                                                    dl.EditControl(
                                                                        id="region-edit-control",
                                                                        draw={
                                                                            "rectangle": {},
                                                                            "polyline": False,
                                                                            "polygon": False,
                                                                            "circle": False,
                                                                            "marker": False,
                                                                            "circlemarker": False,
                                                                        },
                                                                        edit={"edit": False},
                                                                        position="topleft",
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                        style={
                                                            "width": "100%",
                                                            "height": "720px",
                                                            "borderRadius": "8px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="range-panel",
                                                        style={
                                                            "position": "absolute",
                                                            "top": "8px",
                                                            "right": "8px",
                                                            "zIndex": "1000",
                                                            "pointerEvents": "auto",
                                                        },
                                                    ),
                                                ],
                                                style={"position": "relative"},
                                            ),

                                            html.Div(
                                                [
                                                    html.Div(
                                                        id="traffic-frame-label",
                                                        className="history-panel__label",
                                                    ),

                                                    dcc.Slider(
                                                        id="traffic-frame-slider",
                                                        min=0,
                                                        max=1,
                                                        step=1,
                                                        value=0,
                                                        marks=None,
                                                        tooltip={"placement": "bottom"},
                                                    ),

                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "▶ 재생",
                                                                id="traffic-play-btn",
                                                                n_clicks=0,
                                                                className="secondary-button",
                                                                style={"marginRight": "6px"},
                                                            ),
                                                            html.Button(
                                                                "⏭ 초기화",
                                                                id="traffic-reset-btn",
                                                                n_clicks=0,
                                                                className="secondary-button",
                                                            ),
                                                        ],
                                                        style={"marginTop": "8px"},
                                                    ),

                                                    dcc.Interval(
                                                        id="traffic-frame-interval",
                                                        interval=500,
                                                        disabled=True,
                                                        n_intervals=0,
                                                    ),
                                                ],
                                                id="dynamic-frame-wrap",
                                                className="history-panel",
                                                style={"display": "none"},
                                            ),

                                            html.Div(id="run-status", style={"margin": "12px 0", "fontSize": "13px"}),

                                            html.Div(
                                                [
                                                    html.Div(
                                                        id="algo-history-label",
                                                        className="history-panel__label",
                                                    ),
                                                    dcc.Slider(
                                                        id="algo-history-slider",
                                                        min=0,
                                                        max=1,
                                                        step=1,
                                                        value=0,
                                                        marks=None,
                                                        tooltip={"placement": "bottom"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "▶ 재생",
                                                                id="algo-play-btn",
                                                                n_clicks=0,
                                                                className="secondary-button",
                                                                style={"marginRight": "6px"},
                                                            ),
                                                            html.Button(
                                                                "⏭ 초기화",
                                                                id="algo-reset-btn",
                                                                n_clicks=0,
                                                                className="secondary-button",
                                                            ),
                                                        ],
                                                        style={"marginTop": "8px"},
                                                    ),
                                                    dcc.Interval(
                                                        id="algo-frame-interval",
                                                        interval=300,
                                                        disabled=True,
                                                        n_intervals=0,
                                                    ),
                                                ],
                                                id="algo-history-wrap",
                                                className="history-panel",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        id="map-view-wrap",
                                    ),

                                    html.Div(
                                        id="analysis-view-wrap",
                                        className="analysis-view",
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                        ],
                        style={"flex": "1", "padding": "18px", "minWidth": 0},
                    ),

                    html.Div(id="right-resize-handle", className="resize-handle-v"),

                    # ── 우측 Sweep 사이드바 ──────────────────────────────────
                    html.Aside(
                        [
                            html.Button("►", id="right-toggle-btn", n_clicks=0,
                                        className="sidebar-toggle-btn",
                                        style={"textAlign": "left"}),
                            html.Div(
                                [
                                    dcc.Tabs(
                                        id="right-tabs",
                                        value="tab-algo",
                                        children=[
                                            dcc.Tab(
                                                label="알고리즘",
                                                value="tab-algo",
                                                children=[algo_sidebar_layout()],
                                                style={"padding": "10px 4px 0"},
                                                selected_style={"padding": "10px 4px 0",
                                                                "fontWeight": "700"},
                                            ),
                                            dcc.Tab(
                                                label="Sweep",
                                                value="tab-sweep",
                                                children=[
                                                    # ── 모드 선택 ──────────────────────────
                                                    dcc.RadioItems(
                                                        id="sweep-mode",
                                                        options=[
                                                            {"label": "모드 1: 하이퍼파라미터별 성능",
                                                             "value": "mode1"},
                                                            {"label": "모드 2: 알고리즘별 성능",
                                                             "value": "mode2"},
                                                        ],
                                                        value="mode1",
                                                        labelStyle={"display": "block",
                                                                    "fontSize": "12px",
                                                                    "marginBottom": "4px"},
                                                        style={"marginBottom": "14px",
                                                               "padding": "8px",
                                                               "background": "#eef0f3",
                                                               "borderRadius": "12px"},
                                                    ),

                                                    # ── 모드 1 패널 ────────────────────────
                                                    html.Div(
                                                        [
                                                            html.H3("Sweep 설정",
                                                                    className="section-header"),
                                                            html.Div(
                                                                id="sweep-algo-display",
                                                                style={"fontSize": "12px", "color": "#6b7280",
                                                                       "marginTop": "8px",
                                                                       "marginBottom": "8px"},
                                                            ),
                                                            html.Div(id="sweep-params-container"),
                                                            html.Button(
                                                                "Sweep 실행",
                                                                id="sweep-run-btn",
                                                                n_clicks=0,
                                                                className="primary-button",
                                                            ),
                                                            html.Div(
                                                                id="sweep-status",
                                                                style={"fontSize": "12px",
                                                                       "marginTop": "8px"},
                                                            ),

                                                            html.H3("Sweep 결과",
                                                                    className="section-header",
                                                                    style={"marginTop": "var(--sp-base)"}),
                                                            dcc.Graph(
                                                                id="sweep-result-chart",
                                                                style={"height": "300px",
                                                                       "width": "100%",
                                                                       "marginTop": "8px"},
                                                                config={"displayModeBar": False},
                                                                responsive=True,
                                                            ),
                                                            html.Div(
                                                                id="sweep-result-table",
                                                                style={"marginTop": "6px",
                                                                       "maxHeight": "260px",
                                                                       "overflowY": "auto"},
                                                            ),
                                                            html.Button(
                                                                "최적 결과 적용",
                                                                id="sweep-apply-btn",
                                                                n_clicks=0,
                                                                className="primary-button",
                                                                style={"marginTop": "8px"},
                                                            ),
                                                        ],
                                                        id="sweep-mode1-panel",
                                                    ),

                                                    # ── 모드 2 패널 ────────────────────────
                                                    html.Div(
                                                        [
                                                            html.H3("알고리즘 선택 및 하이퍼파라미터",
                                                                    className="section-header"),
                                                            html.Div(
                                                                [_mode2_accordion_item(cls) for cls in REGISTRY],
                                                                id="mode2-accordion",
                                                                style={"marginBottom": "8px"},
                                                            ),

                                                            html.Button(
                                                                "알고리즘 비교 실행",
                                                                id="algo-compare-run-btn",
                                                                n_clicks=0,
                                                                className="primary-button",
                                                                style={"marginTop": "10px",
                                                                       "width": "100%"},
                                                            ),
                                                            html.Div(
                                                                id="algo-compare-status",
                                                                style={"marginTop": "6px",
                                                                       "fontSize": "12px"},
                                                            ),

                                                            html.H3("Sweep 결과",
                                                                    className="section-header",
                                                                    style={"marginTop": "var(--sp-base)"}),
                                                            html.Div(id="algo-compare-results"),
                                                        ],
                                                        id="sweep-mode2-panel",
                                                        style={"display": "none"},
                                                    ),
                                                ],
                                                style={"padding": "10px 4px 0"},
                                                selected_style={"padding": "10px 4px 0",
                                                                "fontWeight": "700"},
                                            ),
                                        ],
                                    ),
                                ],
                                id="right-sidebar-body",
                                className="sidebar-body",
                                style={"overflowY": "auto", "flex": "1", "padding": "16px"},
                            ),
                        ],
                        id="right-sidebar",
                        style={
                            "width": "420px",
                            "minWidth": "420px",
                            "height": "100vh",
                            "display": "flex",
                            "flexDirection": "column",
                            "borderLeft": "1px solid #dedee5",
                            "background": "#ffffff",
                            "boxSizing": "border-box",
                            "transition": "width 0.2s ease, min-width 0.2s ease",
                        },
                    ),
                ],
                style={"display": "flex", "height": "100vh", "overflow": "hidden"},
            ),

            # Region selection popup overlay
            html.Div(
                id="region-popup",
                children=[
                    html.Div(
                        [
                            html.H4("트래픽 영역 설정", style={"margin": "0 0 16px 0", "color": "#111827"}),

                            html.Div(
                                [
                                    html.Label("너비 (km)", style={
                                        "fontWeight": "600",
                                        "fontSize": "13px",
                                        "color": "#374151",
                                    }),
                                    dcc.Input(
                                        id="region-width-km",
                                        type="number",
                                        min=0.1,
                                        max=200,
                                        step=0.01,
                                        style={
                                            "width": "100%",
                                            "padding": "6px",
                                            "borderRadius": "4px",
                                            "border": "1px solid #d1d5db",
                                            "color": "#111827",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),

                            html.Div(
                                [
                                    html.Label("높이 (km)", style={
                                        "fontWeight": "600",
                                        "fontSize": "13px",
                                        "color": "#374151",
                                    }),
                                    dcc.Input(
                                        id="region-height-km",
                                        type="number",
                                        min=0.1,
                                        max=200,
                                        step=0.01,
                                        style={
                                            "width": "100%",
                                            "padding": "6px",
                                            "borderRadius": "4px",
                                            "border": "1px solid #d1d5db",
                                            "color": "#111827",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "20px"},
                            ),

                            html.Div(
                                [
                                    html.Button(
                                        "확인",
                                        id="region-confirm-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "8px 24px",
                                            "marginRight": "8px",
                                            "background": "#7132f5",
                                            "color": "white",
                                            "border": "0",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                        },
                                    ),
                                    html.Button(
                                        "취소",
                                        id="region-cancel-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "8px 24px",
                                            "background": "#6b7280",
                                            "color": "white",
                                            "border": "0",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                        },
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "background": "white",
                            "padding": "24px",
                            "borderRadius": "10px",
                            "boxShadow": "0 8px 32px rgba(0,0,0,0.25)",
                            "minWidth": "280px",
                        },
                    ),
                ],
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100vw",
                    "height": "100vh",
                    "background": "rgba(0,0,0,0.45)",
                    "zIndex": 10000,
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
        ]
    )


# ---------------------------------------------------------------------------
# App / callbacks
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.layout = serve_layout

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Base Station Simulator</title>
        {%favicon%}
        {%css%}
        <style>
            button { cursor: pointer; }
            .leaflet-interactive { cursor: pointer; }
            /* rectangle draw is launched from the sidebar button, not the map toolbar */
            a.leaflet-draw-draw-rectangle { display: none !important; }
            /* Leaflet Draw shows a floating Cancel action while rectangle mode is active. */
            .leaflet-draw-actions { display: none !important; }
            /* delete-layers button is kept enabled for programmatic clear but hidden from UI */
            a.leaflet-draw-edit-remove { display: none !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(
    Output("map-view-wrap", "style"),
    Output("analysis-view-wrap", "style"),
    Input("main-view-mode", "value"),
)
def toggle_main_view(view_mode):
    if view_mode == "analysis":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("analysis-view-wrap", "children"),
    Input("main-view-mode", "value"),
    Input("env-meta", "data"),
    Input("opt-meta", "data"),
    Input("operation-meta", "data"),
    Input("range-meta", "data"),
    Input("sweep-meta", "data"),
    Input("algo-compare-meta", "data"),
    Input("station-specs-store", "data"),
    State("session-id", "data"),
)
def render_analysis_view(
    view_mode,
    env_meta,
    opt_meta,
    _operation_meta,
    range_meta,
    sweep_meta,
    algo_compare_meta,
    station_specs,
    session_id,
):
    if view_mode != "analysis":
        return []

    state = get_session_state(session_id)
    env = state.get("env")
    opt_results = state.get("opt_results")
    operation_results = state.get("operation_results")
    station_count = len((opt_results or {}).get("stations_geo") or [])
    active_mask = operation_active_mask_for_frame(
        operation_results,
        station_count,
        int(getattr(env, "dynamic_frame_index", 0)) if env is not None else 0,
    )
    metrics = (
        compute_frame_metrics(env, opt_results, station_specs, active_mask=active_mask)
        or state.get("opt_stats")
    )
    dynamic_summary = compute_dynamic_scenario_summary(env, opt_results, station_specs, operation_results)
    range_results = state.get("range_results") or []
    sweep_results = state.get("sweep_results") or []
    compare_results = state.get("algo_compare_results") or []

    environment_children = (
        [compact_table(environment_summary_rows(env), page_size=7)]
        if env is not None
        else [analysis_empty("환경 데이터가 없습니다. 영역 지정 후 가상 데이터를 생성해주세요.")]
    )

    metric_children = (
        [compact_table(current_metrics_rows(metrics), page_size=7)]
        if metrics
        else [analysis_empty("계산 실행 후 현재 프레임 성능이 표시됩니다.")]
    )

    dynamic_children: list[Any] = []
    if dynamic_summary:
        dynamic_children.append(compact_table([
            {"항목": "현재 프레임", "값": f"{dynamic_summary['current_frame']} / {dynamic_summary['max_frame']}"},
            {"항목": "평균 커버율", "값": f"{dynamic_summary['avg_traffic_coverage_pct']:.1f}%"},
            {"항목": "최악 커버율", "값": f"{dynamic_summary['worst_traffic_coverage_pct']:.1f}%"},
            {"항목": "최대 부하", "값": format_metric_value("부하", dynamic_summary["max_station_load"])},
        ], page_size=4))
        dynamic_children.append(dcc.Graph(
            figure=dynamic_frame_figure(env, opt_results, station_specs, operation_results),
            config={"displayModeBar": False},
        ))
    else:
        dynamic_children.append(analysis_empty("동적 트래픽 시나리오 결과가 없습니다."))

    optimization_children: list[Any] = []
    if opt_results:
        optimization_children.append(compact_table([
            {"항목": "알고리즘", "값": opt_results.get("algo", "-")},
            {"항목": "Score", "값": format_metric_value("score", opt_results.get("score"))},
            {"항목": "평가 모드", "값": opt_results.get("score_mode", "-")},
            {"항목": "스펙트럼 효율", "값": opt_results.get("spectral_efficiency_mode", "-")},
            {"항목": "k 후보 결과", "값": f"{len(range_results)}개" if range_results else "-"},
        ], page_size=5))
        optimization_children.append(dcc.Graph(
            figure=convergence_figure(opt_results),
            config={"displayModeBar": False},
        ))
    else:
        optimization_children.append(analysis_empty("최적화 결과가 없습니다. 계산 실행 후 표시됩니다."))

    station_children = (
        [compact_table(station_analysis_rows(opt_results, metrics, station_specs), page_size=10)]
        if opt_results
        else [analysis_empty("기지국 결과가 없습니다.")]
    )

    operation_children: list[Any] = []
    if operation_results:
        operation_children.append(compact_table(operation_comparison_rows(operation_results), page_size=10))
        operation_children.append(compact_table(operation_summary_rows(operation_results), page_size=10))
        operation_children.append(dcc.Graph(
            figure=operation_history_figure(operation_results),
            config={"displayModeBar": False},
        ))
    else:
        operation_children.append(analysis_empty("운영 최적화 실행 후 OPEX 결과가 표시됩니다."))

    sweep_children = (
        [compact_table(sweep_summary_rows(sweep_results), page_size=10)]
        if sweep_results
        else [analysis_empty("Sweep 실행 결과가 없습니다.")]
    )

    compare_children = (
        [compact_table(compare_summary_rows(compare_results), page_size=10)]
        if compare_results
        else [analysis_empty("알고리즘 비교 결과가 없습니다.")]
    )

    return [
        html.Div(
            [
                analysis_section("환경 요약", environment_children),
                analysis_section("현재 프레임 성능", metric_children),
                analysis_section("동적 트래픽 시나리오", dynamic_children),
                analysis_section("최적화 결과", optimization_children),
                analysis_section("기지국별 분석", station_children),
                analysis_section("운영 최적화 결과", operation_children),
                analysis_section("Sweep 결과", sweep_children),
                analysis_section("알고리즘 비교", compare_children),
            ],
            className="analysis-grid",
        )
    ]


@app.callback(
    Output("multi-hotspot-controls", "style"),
    Input("traffic-pattern", "value"),
)
def toggle_multi_hotspot_controls(pattern):
    return {"display": "block" if pattern == "multi_hotspot" else "none"}


@app.callback(
    Output("dynamic-traffic-controls", "style"),
    Output("operation-optimization-section", "style"),
    Input("dynamic-traffic", "value"),
)
def toggle_dynamic_controls(dynamic_value):
    dynamic_style = {"display": "block" if normalize_triggered_bool(dynamic_value) else "none"}
    operation_style = {**dynamic_style, "marginTop": "10px"}
    return dynamic_style, operation_style


@app.callback(
    Output("operation-run-btn", "disabled"),
    Input("dynamic-traffic", "value"),
    Input("opt-meta", "data"),
    Input("env-meta", "data"),
)
def toggle_operation_run_button(dynamic_value, opt_meta, env_meta):
    return not (normalize_triggered_bool(dynamic_value) and opt_meta and env_meta)


@app.callback(
    Output("operation-status", "children"),
    Output("operation-meta", "data"),
    Input("operation-run-btn", "n_clicks"),
    State("session-id", "data"),
    State("operation-policy", "value"),
    State({"type": "operation-hyperparam", "name": ALL, "kind": ALL}, "value"),
    State({"type": "operation-hyperparam", "name": ALL, "kind": ALL}, "id"),
    State("station-specs-store", "data"),
    prevent_initial_call=True,
)
def run_operation_optimization(
    n_clicks,
    session_id,
    operation_policy,
    operation_param_values,
    operation_param_ids,
    station_specs,
):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)
    env = state.get("env")
    opt_results = state.get("opt_results")
    operation_params = _parse_operation_hyperparams(operation_param_values, operation_param_ids)
    result = evaluate_operation_optimization(
        env,
        opt_results,
        station_specs,
        normalize_operation_policy(operation_policy),
        operation_params,
    )
    if result is None:
        return (
            html.Span(
                "동적 트래픽 데이터와 기지국 위치 최적화 결과가 필요합니다.",
                style={"color": "#b91c1c", "fontWeight": "600"},
            ),
            no_update,
        )

    state["operation_results"] = result
    if opt_results is not None:
        opt_results["operation_policy"] = result["policy"]
        opt_results["operation_params"] = result.get("operation_params")
    return render_operation_status(result), version_token()


@app.callback(
    Output("synthetic-obstacle-controls", "style"),
    Output("osm-obstacle-controls", "style"),
    Output("geojson-obstacle-controls", "style"),
    Output("geojson-filter-controls", "style"),
    Input("obstacle-source", "value"),
)
def toggle_obstacle_source_controls(source):
    return (
        {"display": "block" if source == "합성" else "none"},
        {"display": "block" if source == "OSM 지도 데이터" else "none"},
        {"display": "block" if source == "GeoJSON 업로드" else "none"},
        {"display": "block" if source == "GeoJSON 업로드" else "none"},
    )





@app.callback(
    Output("create-env-btn", "children"),
    Input("custom-region-store", "data"),
    Input("drawn-region-store", "data"),
    Input("region-draw-activate-dummy", "data"),
)
def update_create_env_button_label(custom_region, drawn_region, is_drawing):
    if is_drawing and not drawn_region and not has_custom_region(custom_region):
        return "취소"
    return "가상 데이터 생성" if has_custom_region(custom_region) else "영역 지정"


@app.callback(
    Output("hyperparam-controls", "children"),
    Input("algo-select", "value"),
)
def render_hyperparam_controls(algo):
    if not algo:
        return []
    optimizer = get_optimizer(algo)
    controls = []
    for p in optimizer.hyperparams or []:
        label = p.label or p.name
        cid = {"type": "hyperparam", "name": p.name, "kind": p.kind}
        controls.append(html.Label(label))
        if p.kind == "int":
            if p.min is not None and p.max is not None and p.step is not None:
                controls.append(dcc.Slider(id=cid, min=int(p.min), max=int(p.max), step=int(p.step), value=int(p.default), tooltip={"placement": "bottom"}))
            else:
                controls.append(dcc.Input(id=cid, type="number", value=int(p.default), style={"width": "100%"}))
        elif p.kind == "float":
            if p.min is not None and p.max is not None:
                controls.append(dcc.Slider(id=cid, min=float(p.min), max=float(p.max), step=float(p.step if p.step is not None else 0.01), value=float(p.default), tooltip={"placement": "bottom"}))
            else:
                controls.append(dcc.Input(id=cid, type="number", value=float(p.default), style={"width": "100%"}))
        elif p.kind == "choice":
            controls.append(dcc.Dropdown(id=cid, options=[{"label": str(x), "value": x} for x in p.choices], value=p.default))
        elif p.kind == "bool":
            controls.append(dcc.Checklist(id=cid, options=[{"label": "사용", "value": "on"}], value=["on"] if bool(p.default) else []))
    return controls


def _operation_param_input(name: str):
    spec = OPERATION_PARAM_SPECS[name]
    input_props = {
        "id": {"type": "operation-hyperparam", "name": name, "kind": "float"},
        "type": "number",
        "value": OPERATION_DEFAULT_PARAMS[name],
        "step": spec["step"],
        "style": {"width": "100%"},
    }
    if "min" in spec:
        input_props["min"] = spec["min"]
    if "max" in spec:
        input_props["max"] = spec["max"]

    return html.Div(
        [
            html.Label(spec["label"], style={"fontSize": "12px", "marginBottom": "2px"}),
            dcc.Input(**input_props),
        ],
        style={"minWidth": 0},
    )


@app.callback(
    Output("operation-hyperparam-controls", "children"),
    Input("operation-policy", "value"),
)
def render_operation_hyperparam_controls(policy):
    policy = normalize_operation_policy(policy)
    policy_params = OPERATION_POLICY_PARAM_NAMES.get(policy, [])

    groups = [
        html.Div(
            [
                html.Div(
                    [_operation_param_input(name) for name in OPERATION_COMMON_PARAM_NAMES],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(118px, 1fr))",
                        "gap": "8px",
                    },
                ),
            ]
        )
    ]

    if policy_params:
        groups.append(
            html.Div(
                [
                    html.Div(
                        [_operation_param_input(name) for name in policy_params],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(118px, 1fr))",
                            "gap": "8px",
                        },
                    ),
                ]
            )
        )

    return html.Div(groups, style={"marginTop": "10px"})


@app.callback(
    Output("noise-caption", "children"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
)
def update_noise_caption(tx_power, path_loss_exp, bandwidth_mhz, sinr_threshold):
    prop = prop_params_base(
        float(path_loss_exp),
        float(bandwidth_mhz),
        float(sinr_threshold),
    )
    r_eff = radius_from_tx(np.asarray([float(tx_power)], dtype=float), prop)[0]

    return (
        f"잡음 바닥: {prop['noise_floor_dbm']:.1f} dBm "
        f"| 단일 기지국 예상 커버 반경: {r_eff:.0f} m (시각화 전용, 실제 커버리지는 SINR 기반)"
    )


@app.callback(
    Output("spectral-eff-wrap", "style"),
    Input("score-mode", "value"),
)
def toggle_spectral_eff_panel(score_mode):
    return {"display": "none", "marginTop": "8px"}




@app.callback(
    Output("area-demand-cell-display", "children"),
    Input("area-demand-mbps-km2", "value"),
    Input("resolution-m", "value"),
)
def update_area_demand_display(area_demand, resolution_m):
    density = safe_float(area_demand, 25.0)
    res = safe_float(resolution_m, 100.0)
    cell_km2 = (res / 1000.0) ** 2
    cell_mbps = density * cell_km2
    return f"셀당 수요: {cell_mbps:.3f} Mbps  (해상도 {int(res)}m 기준)"


@app.callback(
    Output("spec-sliders-container", "children"),
    Output("spec-sliders-wrap", "style"),
    Output("common-spec-wrap", "style"),
    Input("spec-mode", "value"),
    Input("n-stations", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-bandwidth-mhz", "value"),
    State("station-specs-store", "data"),
)
def render_spec_sliders(spec_mode, n_stations, ui_tx_power, ui_bandwidth_mhz, store_data):
    hidden = {"marginTop": "8px", "display": "none"}
    if spec_mode != "기지국별 개별":
        return [], hidden, {"marginTop": "8px"}

    k = safe_int(n_stations, 5)
    default_tx = safe_float(ui_tx_power, 43.0)
    default_bw = safe_float(ui_bandwidth_mhz, 10.0)
    store = store_data or []

    items = []
    for i in range(k):
        saved = store[i] if i < len(store) else {}
        tx_val = safe_float(saved.get("tx_power_dbm"), default_tx)
        bw_val = safe_float(saved.get("bandwidth_mhz"), default_bw)
        items.append(html.Div(
            [
                html.Div(f"기지국 {i + 1}",
                         style={"fontWeight": "600", "fontSize": "12px",
                                "color": "#7132f5", "marginBottom": "4px"}),
                html.Label("송신 전력 (dBm)",
                           style={"fontSize": "11px", "color": "#6b7280"}),
                dcc.Slider(
                    id={"type": "spec-tx-slider", "index": i},
                    min=20, max=50, step=1, value=tx_val,
                    marks={20: "20", 30: "30", 40: "40", 50: "50"},
                    tooltip={"placement": "bottom"},
                ),
                html.Label("대역폭 (MHz)",
                           style={"fontSize": "11px", "color": "#6b7280",
                                  "marginTop": "4px"}),
                dcc.Slider(
                    id={"type": "spec-bw-slider", "index": i},
                    min=1, max=100, step=1, value=bw_val,
                    marks={1: "1", 20: "20", 50: "50", 100: "100"},
                    tooltip={"placement": "bottom"},
                ),
            ],
            style={"marginBottom": "12px", "padding": "8px",
                   "background": "#f7f7f7", "borderRadius": "8px",
                   "border": "1px solid #dedee5"},
        ))

    return items, {"marginTop": "8px", "display": "block"}, {"marginTop": "8px", "display": "none"}


@app.callback(
    Output("station-specs-store", "data"),
    Input({"type": "spec-tx-slider", "index": ALL}, "value"),
    Input({"type": "spec-bw-slider", "index": ALL}, "value"),
    State({"type": "spec-tx-slider", "index": ALL}, "id"),
)
def sync_sliders_to_store(tx_vals, bw_vals, tx_ids):
    if not tx_ids:
        return []
    ordered = sorted(range(len(tx_ids)), key=lambda i: tx_ids[i]["index"])
    return [
        {
            "tx_power_dbm": safe_float(tx_vals[i], 43.0),
            "bandwidth_mhz": safe_float(bw_vals[i], 10.0),
        }
        for i in ordered
    ]


@app.callback(
    Output("create-env-btn", "disabled"),
    Output("env-meta", "data"),
    Output("opt-meta", "data"),
    Output("range-meta", "data"),
    Output("create-status", "children"),
    Input("create-env-btn", "n_clicks"),
    State("session-id", "data"),
    State("sim-map", "bounds"),
    State("sim-map", "center"),
    State("sim-map", "zoom"),
    State("resolution-m", "value"),
    State("traffic-pattern", "value"),
    State("area-demand-mbps-km2", "value"),
    State("base-intensity", "value"),
    State("max-intensity", "value"),
    State("dynamic-traffic", "value"),
    State("dynamic-traffic-type", "value"),
    State("num-hotspots", "value"),
    State("spread-m", "value"),
    State("dynamic-time-steps", "value"),
    State("dynamic-variation", "value"),
    State("dynamic-drift-m", "value"),
    State("obstacle-source", "value"),
    State("obstacle-pattern", "value"),
    State("num-obstacles", "value"),
    State("osm-types", "value"),
    State("geojson-upload", "contents"),
    State("min-obstacle-area-m2", "value"),
    State("max-map-obstacles", "value"),
    State("custom-region-store", "data"),
    State("region-draw-activate-dummy", "data"),
    prevent_initial_call=True,
)
def create_environment(
    n_clicks,
    session_id,
    bounds,
    center,
    zoom,
    resolution_m,
    traffic_pattern,
    area_demand_mbps_km2,
    base_intensity,
    max_intensity,
    dynamic_traffic,
    dynamic_traffic_type,
    num_hotspots,
    spread_m,
    dynamic_time_steps,
    dynamic_variation,
    dynamic_drift_m,
    obstacle_source,
    obstacle_pattern,
    num_obstacles,
    osm_types,
    geojson_contents,
    min_obstacle_area_m2,
    max_map_obstacles,
    custom_region,
    is_drawing,
):
    if not n_clicks:
        raise PreventUpdate

    if not has_custom_region(custom_region):
        if is_drawing:
            return False, no_update, no_update, no_update, ""
        return False, no_update, no_update, no_update, html.Div(
            "지도에서 생성할 영역을 드래그하세요.",
            style={"color": "#b45309", "fontWeight": "600"},
        )

    state = get_session_state(session_id)

    try:
        center_lat = float(custom_region["center_lat"])
        center_lon = float(custom_region["center_lon"])
        width_km = max(float(custom_region["width_km"]), 0.1)
        height_km = max(float(custom_region["height_km"]), 0.1)

        env = SyntheticEnvironment(
            center_lat=center_lat,
            center_lon=center_lon,
            width_km=width_km,
            height_km=height_km,
            resolution_m=safe_float(resolution_m, 100.0),
        )

        is_dynamic = normalize_triggered_bool(dynamic_traffic)

        pattern_params: dict = {}
        if traffic_pattern == "multi_hotspot":
            sigma_cells = max(
                safe_float(spread_m, 300.0) / max(safe_float(resolution_m, 100.0), 1.0),
                1.0,
            )
            pattern_params = {
                "n_centers": safe_int(num_hotspots, 5),
                "sigma_x": sigma_cells,
                "sigma_y": sigma_cells,
            }

        if is_dynamic:
            env.generate_dynamic_traffic_pattern_density(
                area_demand_mbps_km2=safe_float(area_demand_mbps_km2, 150.0),
                pattern=traffic_pattern,
                time_steps=safe_int(dynamic_time_steps, 12),
                variation=safe_float(dynamic_variation, 0.25),
                drift_m=safe_float(dynamic_drift_m, 300.0),
                dynamic_type=dynamic_traffic_type or DEFAULT_DYNAMIC_TRAFFIC_TYPE,
                params=pattern_params,
            )

        else:
            env.generate_traffic_pattern_density(
                area_demand_mbps_km2=safe_float(area_demand_mbps_km2, 150.0),
                pattern=traffic_pattern,
                params=pattern_params,
            )

        selected_osm_types = osm_types or []
        osm_obstacle_types: list[str] = []

        for osm_type in selected_osm_types:
            value = OSM_OBSTACLE_TYPE_VALUES[osm_type]
            if isinstance(value, tuple):
                osm_obstacle_types.extend(value)
            else:
                osm_obstacle_types.append(value)

        osm_obstacle_types = list(dict.fromkeys(osm_obstacle_types))
        uploaded_geojson = decode_upload_to_bytes(geojson_contents)

        applied_count, raw_count = apply_obstacle_source(
            env,
            source=obstacle_source or "합성",
            uploaded_geojson=uploaded_geojson,
            min_area_m2=safe_float(min_obstacle_area_m2, 100.0),
            max_obstacles=safe_int(max_map_obstacles, 100) if max_map_obstacles is not None else None,
            obstacle_pattern=obstacle_pattern or "mixed",
            num_obstacles=safe_int(num_obstacles, 3),
            osm_obstacle_types=osm_obstacle_types,
            append=False,
        )

        state["env"] = env
        state.pop("opt_results", None)
        state.pop("opt_stats", None)
        state.pop("range_results", None)
        state.pop("station_overlay_loads", None)
        state.pop("operation_results", None)

        msg = (
            f"가상 환경 생성 완료 | 영역: {width_km:.2f} km × {height_km:.2f} km | "
            f"{obstacle_source}(장애물): 원본 {raw_count}개 중 {applied_count}개 적용"
        )

        return False, version_token(), None, None, html.Div(
            msg,
            style={"color": "#166534", "fontWeight": "600"},
        )

    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        return (
            False,
            no_update,
            no_update,
            no_update,
            html.Div(
                f"생성 실패: {exc}\n{tb}",
                style={"color": "#b91c1c", "whiteSpace": "pre-wrap"},
            ),
        )


@app.callback(
    Output("dynamic-frame-wrap", "style"),
    Output("traffic-frame-slider", "max"),
    Output("traffic-frame-slider", "value"),
    Output("traffic-frame-label", "children"),
    Input("env-meta", "data"),
    State("session-id", "data"),
)
def refresh_dynamic_frame_controls(env_meta, session_id):
    state = get_session_state(session_id)
    env = state.get("env")
    series = getattr(env, "traffic_series", None) if env is not None else None

    if series is None or getattr(series, "shape", [0])[0] <= 1:
        return {"display": "none"}, 1, 0, ""

    current = int(getattr(env, "dynamic_frame_index", 0))
    max_frame = int(series.shape[0] - 1)
    current = max(0, min(current, max_frame))

    return (
        {"display": "block"},
        max_frame,
        current,
        f"동적 트래픽 프레임: {current} / {max_frame}",
    )


@app.callback(
    Output("traffic-frame-interval", "disabled"),
    Output("traffic-play-btn", "children"),
    Input("traffic-play-btn", "n_clicks"),
    State("traffic-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_traffic_playback(n_clicks, disabled):
    next_disabled = not bool(disabled)
    return next_disabled, "▶ 재생" if next_disabled else "⏸ 일시정지"


@app.callback(
    Output("traffic-frame-slider", "value", allow_duplicate=True),
    Input("traffic-frame-interval", "n_intervals"),
    State("traffic-frame-slider", "value"),
    State("traffic-frame-slider", "max"),
    State("traffic-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def advance_traffic_frame(n_intervals, current_value, max_value, disabled):
    if disabled:
        raise PreventUpdate

    current = safe_int(current_value, 0)
    max_frame = max(0, safe_int(max_value, 0))

    if max_frame <= 0:
        raise PreventUpdate

    return (current + 1) % (max_frame + 1)


@app.callback(
    Output("traffic-frame-slider", "value", allow_duplicate=True),
    Output("traffic-frame-interval", "disabled", allow_duplicate=True),
    Output("traffic-play-btn", "children", allow_duplicate=True),
    Input("traffic-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_traffic_frame(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return 0, True, "▶ 재생"


@app.callback(
    Output("env-meta", "data", allow_duplicate=True),
    Input("traffic-frame-slider", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def set_dynamic_traffic_frame(frame_idx, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None or getattr(env, "traffic_series", None) is None:
        raise PreventUpdate

    max_frame = int(env.traffic_series.shape[0] - 1)
    frame = max(0, min(safe_int(frame_idx, 0), max_frame))
    env.set_traffic_frame(frame)
    state["env"] = env

    return version_token()


@app.callback(
    Output("overlay-layers", "children"),
    Input("env-meta", "data"),
    Input("opt-meta", "data"),
    Input("station-specs-store", "data"),
    Input("spec-mode", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
    Input("ui-max-coord", "value"),
    Input("selected-station", "data"),
    Input("map-layer-mode", "value"),
    Input("custom-region-store", "data"),
    Input("algo-history-store", "data"),
    Input("algo-history-slider", "value"),
    Input("opt-live-store", "data"),
    Input("operation-meta", "data"),
    State("session-id", "data"),
)
def update_map_layers(
    env_meta,
    opt_meta,
    station_specs,
    spec_mode,
    ui_tx_power,
    ui_path_loss_exp,
    ui_bandwidth_mhz,
    ui_sinr_threshold,
    ui_max_coord,
    selected_station_idx,
    map_layer_mode,
    custom_region,
    algo_history,
    history_frame_idx,
    opt_live,
    operation_meta,
    session_id,
):
    children: list = []

    # Show confirmed custom region boundary
    if isinstance(custom_region, dict):
        try:
            sw = [custom_region["center_lat"] - custom_region["height_km"] / 2 / 110.574,
                  custom_region["center_lon"] - custom_region["width_km"] / 2 / (111.32 * np.cos(np.radians(custom_region["center_lat"])))]
            ne = [custom_region["center_lat"] + custom_region["height_km"] / 2 / 110.574,
                  custom_region["center_lon"] + custom_region["width_km"] / 2 / (111.32 * np.cos(np.radians(custom_region["center_lat"])))]
            children.append(
                dl.Rectangle(
                    bounds=[sw, ne],
                    pathOptions={"color": "#f59e0b", "weight": 2, "fillOpacity": 0.05, "dashArray": "6 4"},
                )
            )
        except (KeyError, TypeError, ZeroDivisionError):
            pass

    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        return children

    opt_results = state.get("opt_results")
    opt_stats = state.get("opt_stats")
    opt_results, station_specs = live_visualization_state(
        opt_results,
        spec_mode,
        station_specs,
        ui_tx_power,
        ui_path_loss_exp,
        ui_bandwidth_mhz,
        ui_sinr_threshold,
        ui_max_coord,
    )
    operation_results = state.get("operation_results")
    station_count = len((opt_results or {}).get("stations_geo") or [])
    active_mask = operation_active_mask_for_frame(
        operation_results,
        station_count,
        int(getattr(env, "dynamic_frame_index", 0)),
    )

    df = env_dataframe_for_current_frame(env)
    status_list, overlay_loads, sinr_per_cell = compute_status_overlay(
        env,
        df,
        opt_results,
        opt_stats,
        station_specs,
        active_mask=active_mask,
    )
    state["station_overlay_loads"] = overlay_loads

    traffic_geojson = build_traffic_geojson(
        env,
        df,
        map_layer_mode,
        status_list,
        sinr_per_cell=sinr_per_cell if (opt_results and opt_stats) else None,
    )

    traffic_options = {
        "style": TRAFFIC_STYLE,
        "onEachFeature": TRAFFIC_ON_EACH_FEATURE,
    }

    children.append(
        dl.GeoJSON(
            id="traffic-geojson",
            data=traffic_geojson,
            options=traffic_options,
        )
    )

    # Live optimization preview (during background thread execution)
    live_progress = state.get("opt_progress", {})
    if live_progress.get("running") and live_progress.get("stations_geo"):
        for i, (lat, lon) in enumerate(live_progress["stations_geo"]):
            children.append(
                dl.Marker(
                    position=[lat, lon],
                    children=[dl.Tooltip(f"진행 중 #{i + 1}")],
                )
            )
        return children

    # History replay: show intermediate station positions (orange markers)
    history_active = False
    if isinstance(algo_history, dict) and algo_history.get("frames"):
        frames = algo_history["frames"]
        n_frames = len(frames)
        idx = min(safe_int(history_frame_idx, 0), n_frames - 1)
        if idx < n_frames - 1:
            history_active = True
            frame = frames[idx]
            stations_geo = frame.get("stations_geo", [])

            # Fading trail from previous snapshots
            trail_start = max(0, idx - 4)
            for ti in range(trail_start, idx):
                alpha = 0.15 + 0.15 * (ti - trail_start + 1)
                for lat, lon in frames[ti].get("stations_geo", []):
                    children.append(
                        dl.CircleMarker(
                            center=[lat, lon],
                            radius=5,
                            pathOptions={
                                "color": "#6b7280",
                                "fillColor": "#9ca3af",
                                "fillOpacity": alpha,
                                "weight": 1,
                            },
                        )
                    )

            # Current frame stations (icon markers)
            for i, (lat, lon) in enumerate(stations_geo):
                children.append(
                    dl.Marker(
                        position=[lat, lon],
                        children=[dl.Tooltip(f"Station #{i + 1}")],
                    )
                )

    if not history_active and opt_results and opt_stats:
        children.extend(build_station_circles(
            opt_results, opt_stats, station_specs,
            selected_station_idx if isinstance(selected_station_idx, int) else None,
            overlay_loads,
            active_mask=active_mask,
        ))

    return children



@app.callback(
    Output("station-layer", "children"),
    Input("env-meta", "data"),
    Input("opt-meta", "data"),
    Input("operation-meta", "data"),
    Input("spec-mode", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
    Input("ui-max-coord", "value"),
    Input("selected-station", "data"),
    Input("algo-history-store", "data"),
    Input("algo-history-slider", "value"),
    State("station-specs-store", "data"),
    State("session-id", "data"),
)
def update_station_markers(
    env_meta,
    opt_meta,
    operation_meta,
    spec_mode,
    ui_tx_power,
    ui_path_loss_exp,
    ui_bandwidth_mhz,
    ui_sinr_threshold,
    ui_max_coord,
    selected_station_idx,
    algo_history,
    history_frame_idx,
    station_specs,
    session_id,
):
    # 히스토리 재생 중(마지막 프레임 아님)이면 최종 결과 마커 숨김
    if isinstance(algo_history, dict) and algo_history.get("frames"):
        frames = algo_history["frames"]
        idx = min(safe_int(history_frame_idx, 0), len(frames) - 1)
        if idx < len(frames) - 1:
            return []

    state = get_session_state(session_id)
    opt_results = state.get("opt_results")
    opt_stats = state.get("opt_stats")
    if not opt_results or not opt_stats:
        return []
    opt_results, station_specs = live_visualization_state(
        opt_results,
        spec_mode,
        station_specs,
        ui_tx_power,
        ui_path_loss_exp,
        ui_bandwidth_mhz,
        ui_sinr_threshold,
        ui_max_coord,
    )
    env = state.get("env")
    operation_results = state.get("operation_results")
    station_count = len(opt_results.get("stations_geo") or [])
    active_mask = operation_active_mask_for_frame(
        operation_results,
        station_count,
        int(getattr(env, "dynamic_frame_index", 0)) if env is not None else 0,
    )
    if env is not None:
        df = env_dataframe_for_current_frame(env)
        _status, operation_loads, _sinr = compute_status_overlay(
            env,
            df,
            opt_results,
            opt_stats,
            station_specs,
            active_mask=active_mask,
        )
    else:
        operation_loads = np.zeros(0)
    frame_stats = compute_frame_metrics(env, opt_results, station_specs)
    overlay_loads = (
        operation_loads
        if active_mask is not None and len(operation_loads) > 0
        else
        np.asarray(frame_stats.get("station_loads"), dtype=float)
        if frame_stats and frame_stats.get("station_loads") is not None
        else state.get("station_overlay_loads", np.zeros(0))
    )
    return build_station_markers(
        opt_results, opt_stats, station_specs,
        selected_station_idx if isinstance(selected_station_idx, int) else None,
        overlay_loads,
        active_mask=active_mask,
    )


@app.callback(
    Output({"type": "station-popup-load", "index": ALL}, "children"),
    Input("env-meta", "data"),
    Input("operation-meta", "data"),
    Input("station-specs-store", "data"),
    Input("spec-mode", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
    Input("ui-max-coord", "value"),
    State({"type": "station-popup-load", "index": ALL}, "id"),
    State("session-id", "data"),
)
def update_station_popup_loads(
    env_meta,
    operation_meta,
    station_specs,
    spec_mode,
    ui_tx_power,
    ui_path_loss_exp,
    ui_bandwidth_mhz,
    ui_sinr_threshold,
    ui_max_coord,
    load_ids,
    session_id,
):
    if not load_ids:
        raise PreventUpdate

    state = get_session_state(session_id)
    opt_results = state.get("opt_results")
    opt_stats = state.get("opt_stats")
    env = state.get("env")
    if env is None or not opt_results:
        return [no_update for _ in load_ids]
    opt_results, station_specs = live_visualization_state(
        opt_results,
        spec_mode,
        station_specs,
        ui_tx_power,
        ui_path_loss_exp,
        ui_bandwidth_mhz,
        ui_sinr_threshold,
        ui_max_coord,
    )

    station_count = len(opt_results.get("stations_geo") or [])
    active_mask = operation_active_mask_for_frame(
        state.get("operation_results"),
        station_count,
        int(getattr(env, "dynamic_frame_index", 0)),
    )
    if active_mask is not None and opt_stats:
        df = env_dataframe_for_current_frame(env)
        _status, loads, _sinr = compute_status_overlay(
            env,
            df,
            opt_results,
            opt_stats,
            station_specs,
            active_mask=active_mask,
        )
    else:
        metrics = compute_frame_metrics(env, opt_results, station_specs)
        if not metrics:
            return [no_update for _ in load_ids]
        loads = np.asarray(metrics.get("station_loads", []), dtype=float)

    values = []
    for id_obj in load_ids:
        idx = safe_int(id_obj.get("index") if isinstance(id_obj, dict) else None, -1)
        active = True if active_mask is None or idx < 0 or idx >= len(active_mask) else bool(active_mask[idx])
        load = float(loads[idx]) if active and 0 <= idx < len(loads) else 0.0
        status_suffix = f" | {'ON' if active else 'SLEEP/OFF'}" if active_mask is not None else ""
        values.append(f"Load: {load:.1f}{status_suffix}")
    return values


@app.callback(
    Output("station-specs-store", "data", allow_duplicate=True),
    Input({"type": "station-tx-input", "index": ALL}, "value"),
    Input({"type": "station-bandwidth-input", "index": ALL}, "value"),
    State({"type": "station-tx-input", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def realtime_popup_to_store(tx_vals, bw_vals, tx_ids):
    if not tx_ids:
        raise PreventUpdate
    ordered = sorted(range(len(tx_ids)), key=lambda i: tx_ids[i]["index"])
    return [
        {
            "tx_power_dbm": safe_float(tx_vals[i], 43.0),
            "bandwidth_mhz": safe_float(bw_vals[i] if bw_vals and i < len(bw_vals) else None, 10.0),
        }
        for i in ordered
    ]


# ---------------------------------------------------------------------------
# Optimization: helpers + background thread + callbacks
# ---------------------------------------------------------------------------

def _parse_hyperparams(hp_values, hp_ids, hp_defaults: dict) -> dict[str, Any]:
    """UI hyperparam widgets → typed dict."""
    hyperparams: dict[str, Any] = {}
    for value, id_obj in zip(hp_values or [], hp_ids or []):
        name = id_obj.get("name")
        kind = id_obj.get("kind")
        default = hp_defaults.get(name, 0)
        if kind == "bool":
            hyperparams[name] = bool(value) if value is not None else bool(default)
        elif kind == "int":
            hyperparams[name] = safe_int(value, int(default))
        elif kind == "float":
            hyperparams[name] = safe_float(value, float(default))
        else:
            hyperparams[name] = value if value is not None else default
    return hyperparams


def _build_k_list(n_stations) -> list[int]:
    return [safe_int(n_stations, 5)]


def _run_optimization_thread(
    session_id: str,
    algo: str,
    hyperparams: dict,
    k_list: list[int],
    prop: dict,
    spec_mode: str,
    station_specs,
    ui_tx_power,
    score_mode: str = "traffic",
    spectral_efficiency_mode: str = "shannon",
    weight_scale: float = 1.0,
    operation_policy: str = DEFAULT_OPERATION_POLICY,
) -> None:
    """백그라운드 스레드: 최적화 실행 후 세션 상태에 결과 저장."""
    try:
        state = get_session_state(session_id)
        env = state.get("env")
        if env is None:
            state["opt_progress"] = {"running": False, "done": False,
                                     "error": "env가 없습니다. 먼저 데이터를 생성하세요."}
            return

        start_time = time.time()
        optimizer = get_optimizer(algo)
        range_results = []
        operation_policy = normalize_operation_policy(operation_policy)

        for k_idx, k in enumerate(k_list):
            tx_k = tx_power_for_k(
                k,
                safe_float(ui_tx_power, 43.0),
                spec_mode,
                station_specs,
            )
            radius_k = radius_from_tx(tx_k, prop)
            problem = ProblemInput.from_env(
                env,
                radius_m=radius_k,
                capacity=np.full(k, 1e10),
                station_candidate_points=env.station_candidate_points,
                path_loss_exponent=prop["path_loss_exponent"],
                path_loss_ref_db=prop["path_loss_ref_db"],
                tx_power_dbm=tx_k,
                noise_floor_dbm=prop["noise_floor_dbm"],
                sinr_threshold_db=prop["sinr_threshold_db"],
                bandwidth_mhz=prop["bandwidth_mhz"],
                score_mode=score_mode,
                spectral_efficiency_mode=spectral_efficiency_mode,
                weight_scale=weight_scale,
                interference_threshold_dbm=prop["noise_floor_dbm"],
                max_coord_stations=prop.get("max_coord_stations", 1),
            )

            def _progress_cb(it, total, best_stations_local, best_score,
                             _k_idx=k_idx, _problem=problem):
                geo = convert_to_geo(best_stations_local, _problem)
                live_entry = {
                    "iter": int(it),
                    "best_score": float(best_score),
                    "gen_score": float(best_score),
                }
                live_history = state.setdefault("opt_live_history", [])
                if live_history and live_history[-1].get("iter") == live_entry["iter"]:
                    live_history[-1] = live_entry
                else:
                    live_history.append(live_entry)
                state["opt_progress"] = {
                    "running": True, "done": False, "error": None,
                    "algo": algo,
                    "k_current": _k_idx + 1, "k_total": len(k_list),
                    "iter": int(it), "total": int(total),
                    "best_score": float(best_score),
                    "stations_geo": geo.tolist(),
                    "score_series": live_history.copy(),
                }

            result = optimizer.optimize(problem, n_stations=k,
                                        callback=_progress_cb, **hyperparams)

            stations_geo = convert_to_geo(result.stations, problem)
            stations_df = pd.DataFrame(stations_geo, columns=["lat", "lon"])
            stats_out = dict(result.metrics)
            stats_out["n_stations"] = k
            stats_out["total_tx_power_w"] = float(np.sum(10 ** ((tx_k - 30) / 10)))
            result_pack = {
                "k": k,
                "score": float(result.score),
                "covered_traffic": float(result.metrics.get("covered_traffic", 0)),
                "covered_area": float(result.metrics.get("covered_area", 0)),
                "opt_results": {
                    "algo": algo,
                    "score": float(result.score),
                    "stations_geo": stations_df.to_dict("records"),
                    "history": result.history,
                    "prop_params": {**prop, "tx_power_dbm": tx_k.tolist()},
                    "score_mode": score_mode,
                    "spectral_efficiency_mode": spectral_efficiency_mode,
                    "weight_scale": weight_scale,
                    "operation_policy": operation_policy,
                },
                "stats": stats_out,
            }
            range_results.append(result_pack)

        best_res = max(range_results, key=lambda x: x["score"])
        best_opt = best_res["opt_results"]
        best_stats = best_res["stats"]
        elapsed = time.time() - start_time
        _opt_logger.info("opt_thread done: algo=%s best_k=%s score=%.4f elapsed=%.2fs",
                         algo, best_res["k"], best_res["score"], elapsed)

        # 결과를 먼저 저장한 뒤 done=True 신호 (순서 보장)
        state["range_results"] = range_results
        state["opt_results"] = best_opt
        state["opt_stats"] = best_stats
        state.pop("operation_results", None)
        state["opt_progress"] = {
            "running": False, "done": True, "error": None,
            "elapsed": elapsed,
            "best_k": best_res["k"],
            "best_score": best_res["score"],
            "k_total": len(k_list),
        }

    except Exception:
        tb = traceback.format_exc(limit=6)
        _opt_logger.error("opt_thread FAILED: algo=%s\n%s", algo, tb)
        try:
            state = get_session_state(session_id)
            state["opt_progress"] = {"running": False, "done": False, "error": tb}
        except Exception:
            pass


def _make_progress_html(algo: str, k_cur: int, k_tot: int,
                        it: int, total: int, best_score: float) -> html.Div:
    if total > 0:
        pct = min(100.0, it / total * 100)
        overall_pct = ((k_cur - 1) / k_tot + pct / 100.0 / k_tot) * 100.0
        label = f"[{algo}] k {k_cur}/{k_tot} · iter {it}/{total} ({pct:.0f}%) · score {best_score:.2f}"
    else:
        overall_pct = 50.0
        label = f"[{algo}] k {k_cur}/{k_tot} · 계산 중... · score {best_score:.2f}"
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "marginBottom": "4px"}),
            html.Div(
                html.Div(
                    style={
                        "height": "8px",
                        "width": f"{overall_pct:.1f}%",
                        "background": "#7132f5",
                        "borderRadius": "4px",
                        "transition": "width 0.4s ease",
                    }
                ),
                style={
                    "background": "#dedee5",
                    "borderRadius": "4px",
                    "overflow": "hidden",
                    "height": "8px",
                },
            ),
        ],
        style={"marginTop": "4px"},
    )


@app.callback(
    Output("optimize-btn", "disabled"),
    Output("opt-poll-interval", "disabled"),
    Output("run-status", "children"),
    Output("stats-panel", "children", allow_duplicate=True),
    Output("opt-meta", "data", allow_duplicate=True),
    Output("opt-live-store", "data", allow_duplicate=True),
    Input("optimize-btn", "n_clicks"),
    State("session-id", "data"),
    State("algo-select", "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "id"),
    State("n-stations", "value"),
    State("spec-mode", "value"),
    State("station-specs-store", "data"),
    State("ui-tx-power", "value"),
    State("ui-path-loss-exp", "value"),
    State("ui-bandwidth-mhz", "value"),
    State("ui-sinr-threshold", "value"),
    State("ui-max-coord", "value"),
    State("score-mode", "value"),
    State("spectral-eff-mode", "value"),
    State("operation-policy", "value"),
    prevent_initial_call=True,
)
def start_optimization_job(
    n_clicks, session_id, algo,
    hp_values, hp_ids,
    n_stations,
    spec_mode, station_specs,
    ui_tx_power, ui_path_loss_exp, ui_bandwidth_mhz, ui_sinr_threshold, ui_max_coord,
    score_mode, spectral_eff_mode,
    operation_policy,
):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)

    if state.get("opt_progress", {}).get("running"):
        return False, True, html.Div("이미 계산 중입니다.", style={"color": "#cf202f"}), no_update, no_update, no_update

    if state.get("env") is None:
        return False, True, html.Div("먼저 데이터를 생성해주세요.", style={"color": "#b91c1c"}), no_update, no_update, no_update

    optimizer = get_optimizer(algo)
    hp_defaults = {p.name: p.default for p in optimizer.hyperparams}
    hyperparams = _parse_hyperparams(hp_values, hp_ids, hp_defaults)
    k_list = _build_k_list(n_stations)
    prop = prop_params_base(
        path_loss_exponent=safe_float(ui_path_loss_exp, 3.5),
        bandwidth_mhz=safe_float(ui_bandwidth_mhz, 10.0),
        sinr_threshold_db=safe_float(ui_sinr_threshold, 3.0),
        max_coord_stations=safe_int(ui_max_coord, 1),
    )

    _opt_logger.info("opt_job start: algo=%s k_list=%s hp=%s", algo, k_list, hyperparams)

    state["opt_progress"] = {
        "running": True, "done": False, "error": None,
        "algo": algo,
        "k_current": 1, "k_total": len(k_list),
        "iter": 0, "total": 0, "best_score": 0.0, "stations_geo": [],
        "score_series": [],
    }
    state["opt_live_history"] = []

    threading.Thread(
        target=_run_optimization_thread,
        args=(session_id, algo, hyperparams, k_list, prop,
              spec_mode, station_specs,
              ui_tx_power,
              score_mode or "traffic",
              spectral_eff_mode or "shannon",
              1.0,
              normalize_operation_policy(operation_policy)),
        daemon=True,
    ).start()

    state.pop("opt_stats", None)
    state.pop("opt_results", None)
    state.pop("operation_results", None)

    status = _make_progress_html(algo, 1, len(k_list), 0, 0, 0.0)
    return True, False, status, _empty_stats_cards(), version_token(), None


@app.callback(
    Output("run-status", "children", allow_duplicate=True),
    Output("opt-live-store", "data"),
    Output("opt-meta", "data", allow_duplicate=True),
    Output("range-meta", "data", allow_duplicate=True),
    Output("optimize-btn", "disabled", allow_duplicate=True),
    Output("opt-poll-interval", "disabled", allow_duplicate=True),
    Input("opt-poll-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_optimization_progress(n_intervals, session_id):
    state = get_session_state(session_id)
    progress = state.get("opt_progress")

    if not progress:
        raise PreventUpdate

    # Error branch
    if progress.get("error") and not progress.get("running"):
        tb = progress["error"]
        state["opt_progress"] = {}
        state.pop("opt_live_history", None)
        return (
            html.Div(f"계산 실패:\n{tb}",
                     style={"color": "#b91c1c", "whiteSpace": "pre-wrap"}),
            None, no_update, no_update,
            False, True,
        )

    # Done branch
    if progress.get("done"):
        elapsed = progress.get("elapsed", 0.0)
        k_total = progress.get("k_total", 1)
        best_k = progress.get("best_k", "?")
        best_score = progress.get("best_score", 0.0)
        state["opt_progress"] = {}
        _opt_logger.info("poll: done — best_k=%s score=%.4f elapsed=%.2fs",
                         best_k, best_score, elapsed)
        status = html.Div(
            f"계산 완료: {k_total}개 시나리오, 최고 k={best_k}, "
            f"score={best_score:.2f}, 소요 {elapsed:.2f}초",
            style={"color": "#166534"},
        )
        state.pop("opt_live_history", None)
        return status, None, version_token(), version_token(), False, True

    # Running branch
    if not progress.get("running"):
        raise PreventUpdate

    algo = progress.get("algo", "")
    k_cur = progress.get("k_current", 1)
    k_tot = progress.get("k_total", 1)
    it = progress.get("iter", 0)
    total = progress.get("total", 0)
    best_score = progress.get("best_score", 0.0)
    live_data = {
        "running": True,
        "algo": algo,
        "score_series": progress.get("score_series") or state.get("opt_live_history", []),
        "updated_at": version_token(),
    }

    status = _make_progress_html(algo, k_cur, k_tot, it, total, best_score)
    return status, live_data, no_update, no_update, no_update, no_update


@app.callback(
    Output("stats-panel", "children"),
    Input("opt-meta", "data"),
    Input("env-meta", "data"),
    Input("operation-meta", "data"),
    Input("station-specs-store", "data"),
    Input("spec-mode", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
    Input("ui-max-coord", "value"),
    State("session-id", "data"),
)
def render_stats_panel(
    opt_meta,
    env_meta,
    _operation_meta,
    station_specs,
    spec_mode,
    ui_tx_power,
    ui_path_loss_exp,
    ui_bandwidth_mhz,
    ui_sinr_threshold,
    ui_max_coord,
    session_id,
):
    state = get_session_state(session_id)
    env = state.get("env")
    opt_results = state.get("opt_results")
    opt_results, station_specs = live_visualization_state(
        opt_results,
        spec_mode,
        station_specs,
        ui_tx_power,
        ui_path_loss_exp,
        ui_bandwidth_mhz,
        ui_sinr_threshold,
        ui_max_coord,
    )
    base_stats = state.get("opt_stats")
    operation_results = state.get("operation_results")
    station_count = len((opt_results or {}).get("stations_geo") or [])
    active_mask = operation_active_mask_for_frame(
        operation_results,
        station_count,
        int(getattr(env, "dynamic_frame_index", 0)) if env is not None else 0,
    )
    stats = compute_frame_metrics(env, opt_results, station_specs, active_mask=active_mask) or base_stats

    if not stats:
        return _empty_stats_cards()

    summary = compute_dynamic_scenario_summary(env, opt_results, station_specs, operation_results)

    total_t = float(stats.get("total_traffic", 0))
    cov_t = float(stats.get("covered_traffic", 0))
    total_a = float(stats.get("total_area", 0))
    cov_a = float(stats.get("covered_area", 0))

    traffic_cov_pct = (cov_t / total_t) * 100 if total_t > 0 else 0
    area_cov_pct = (cov_a / total_a) * 100 if total_a > 0 else 0

    mean_sinr = stats.get("mean_sinr_db")

    total_tp = float(stats.get("total_throughput_mbps", 0.0))
    total_tx_w = stats.get("total_tx_power_w")
    if total_tx_w and total_tx_w > 0:
        energy_eff = total_tp / total_tx_w
        energy_eff_str = f"{energy_eff:.3f} Mbps/W"
    else:
        energy_eff_str = "-"

    # 트래픽이 Mbps 단위면 소수, 추상 단위면 정수로 표시
    t_fmt = (lambda v: f"{v:.2f} Mbps") if total_t < 1e4 else (lambda v: f"{int(v)}")
    if stats.get("operation_active_mask_applied"):
        station_count_value = (
            f"{int(stats.get('active_station_count', stats.get('n_stations', 0)))} / "
            f"{int(stats.get('total_station_count', stats.get('n_stations', 0)))} 활성"
        )
    else:
        station_count_value = f"{stats.get('n_stations', '-')}"

    cards = []

    cards.extend([
        metric_card("총 트래픽", t_fmt(total_t)),
        metric_card("커버된 트래픽", f"{t_fmt(cov_t)} ({traffic_cov_pct:.1f}%)"),
        metric_card("커버된 면적", f"{int(cov_a)} 격자 ({area_cov_pct:.1f}%)"),
        metric_card("평균 SINR", f"{mean_sinr:.1f} dB" if mean_sinr is not None else "-"),
        metric_card("총 처리량", f"{total_tp:.1f} Mbps"),
        metric_card("기지국 수", station_count_value),
        metric_card("에너지 효율", energy_eff_str),
    ])

    if summary:
        cards.extend([
            metric_card("평균 커버율", f"{summary['avg_traffic_coverage_pct']:.1f}%"),
            metric_card("최악 커버율", f"{summary['worst_traffic_coverage_pct']:.1f}%"),
            metric_card("최대 부하", t_fmt(float(summary["max_station_load"]))),
        ])

    return cards


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Input("apply-k-btn", "n_clicks"),
    State("range-k-dropdown", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_range_selection(n_clicks, selected_k, session_id):
    if not n_clicks or selected_k is None:
        raise PreventUpdate

    state = get_session_state(session_id)
    results = state.get("range_results") or []

    selected = next((r for r in results if int(r["k"]) == int(selected_k)), None)

    if selected is None:
        raise PreventUpdate

    state["opt_results"] = selected["opt_results"]
    state["opt_stats"] = selected["stats"]
    state.pop("operation_results", None)

    return (
        version_token(),
        html.Div(f"k={selected_k} 결과를 지도에 적용했습니다.", style={"color": "#166534"}),
    )


@app.callback(
    Output("download-gis-csv", "data"),
    Input("download-gis-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_gis_csv(n_clicks, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        raise PreventUpdate

    df = env.get_dataframe()
    return dcc.send_data_frame(df.to_csv, "traffic_geo.csv", index=False)


@app.callback(
    Output("download-local-csv", "data"),
    Input("download-local-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_local_csv(n_clicks, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        raise PreventUpdate

    local_data = env.get_local_data_top_left()
    df = pd.DataFrame(local_data, columns=["x", "y", "traffic"])

    return dcc.send_data_frame(df.to_csv, "traffic_local.csv", index=False)


# ---------------------------------------------------------------------------
# Region selection callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("drawn-region-store", "data"),
    Input("region-edit-control", "geojson"),
    prevent_initial_call=True,
)
def handle_drawn_region(geojson):
    """Capture bounding box of the most recently drawn rectangle."""
    if not geojson or not isinstance(geojson, dict):
        raise PreventUpdate
    features = geojson.get("features", [])
    if not features:
        raise PreventUpdate
    feature = features[-1]
    coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
    if not coords or len(coords) < 3:
        raise PreventUpdate
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    south, north = float(min(lats)), float(max(lats))
    west, east = float(min(lons)), float(max(lons))
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    width_km = geodesic((south, west), (south, east)).km
    height_km = geodesic((south, west), (north, west)).km
    return {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "width_km": round(width_km, 3),
        "height_km": round(height_km, 3),
    }


@app.callback(
    Output("region-popup", "style"),
    Output("region-width-km", "value"),
    Output("region-height-km", "value"),
    Input("drawn-region-store", "data"),
)
def show_region_popup(region_data):
    """Show/hide the dimension-adjustment popup based on drawn region."""
    base_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "rgba(0,0,0,0.45)",
        "zIndex": 10000,
        "alignItems": "center",
        "justifyContent": "center",
    }
    if not region_data or not isinstance(region_data, dict):
        return {**base_style, "display": "none"}, no_update, no_update
    w = round(region_data.get("width_km", 2.0), 2)
    h = round(region_data.get("height_km", 2.0), 2)
    return {**base_style, "display": "flex"}, w, h


@app.callback(
    Output("custom-region-store", "data"),
    Output("drawn-region-store", "data", allow_duplicate=True),
    Output("editcontrol-clear-count", "data"),
    Output("custom-region-info", "children"),
    Output("custom-region-info", "style"),
    Output("clear-region-btn", "style"),
    Output("create-status", "children", allow_duplicate=True),
    Output("region-draw-activate-dummy", "data", allow_duplicate=True),
    Input("region-confirm-btn", "n_clicks"),
    State("drawn-region-store", "data"),
    State("region-width-km", "value"),
    State("region-height-km", "value"),
    State("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def apply_region(n_clicks, region_data, width_km, height_km, clear_count):
    """Store confirmed custom region and clear the temporary drawn shape."""
    if not n_clicks or not isinstance(region_data, dict):
        raise PreventUpdate
    w = safe_float(width_km, region_data.get("width_km", 2.0))
    h = safe_float(height_km, region_data.get("height_km", 2.0))
    w = max(w, 0.1)
    h = max(h, 0.1)
    custom = {
        "center_lat": region_data["center_lat"],
        "center_lon": region_data["center_lon"],
        "width_km": w,
        "height_km": h,
    }
    info_text = f"선택 영역: {w:.2f} km × {h:.2f} km"
    info_style = {
        "display": "block",
        "fontSize": "12px",
        "marginTop": "8px",
        "padding": "6px 8px",
        "background": "rgba(20, 158, 97, 0.10)",
        "border": "1px solid #86efac",
        "borderRadius": "4px",
        "color": "#166534",
    }
    clear_style = {
        "display": "block",
        "width": "100%",
        "padding": "6px 12px",
        "marginTop": "4px",
        "cursor": "pointer",
        "background": "#cf202f",
        "color": "white",
        "border": "0",
        "borderRadius": "6px",
        "fontSize": "12px",
        "fontWeight": "600",
    }
    return custom, None, int(clear_count or 0) + 1, info_text, info_style, clear_style, "", False


@app.callback(
    Output("drawn-region-store", "data", allow_duplicate=True),
    Output("editcontrol-clear-count", "data", allow_duplicate=True),
    Output("region-draw-activate-dummy", "data", allow_duplicate=True),
    Input("region-cancel-btn", "n_clicks"),
    State("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def cancel_region(n_clicks, clear_count):
    """Dismiss the popup and clear the drawn shape."""
    if not n_clicks:
        raise PreventUpdate
    return None, int(clear_count or 0) + 1, False


@app.callback(
    Output("region-edit-control", "editToolbar"),
    Input("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def sync_editcontrol_clear(count):
    """Programmatically clear all drawn shapes whenever the counter increments."""
    if not count:
        raise PreventUpdate
    return {"mode": "remove", "action": "clear all", "n_clicks": int(count)}


@app.callback(
    Output("custom-region-store", "data", allow_duplicate=True),
    Output("custom-region-info", "children", allow_duplicate=True),
    Output("custom-region-info", "style", allow_duplicate=True),
    Output("clear-region-btn", "style", allow_duplicate=True),
    Output("env-meta", "data", allow_duplicate=True),
    Output("opt-meta", "data", allow_duplicate=True),
    Output("range-meta", "data", allow_duplicate=True),
    Output("opt-live-store", "data", allow_duplicate=True),
    Output("algo-history-store", "data", allow_duplicate=True),
    Output("selected-station", "data", allow_duplicate=True),
    Output("create-status", "children", allow_duplicate=True),
    Input("clear-region-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def clear_custom_region(n_clicks, session_id):
    """Remove the confirmed custom region and any generated data for it."""
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)
    for key in (
        "env",
        "opt_results",
        "opt_stats",
        "range_results",
        "station_overlay_loads",
        "opt_progress",
        "sweep_config",
        "sweep_results",
        "sweep_progress",
        "algo_compare_config",
        "algo_compare_results",
        "algo_compare_progress",
    ):
        state.pop(key, None)

    hidden_info = {
        "display": "none",
        "fontSize": "12px",
        "marginTop": "8px",
        "padding": "6px 8px",
        "background": "rgba(20, 158, 97, 0.10)",
        "border": "1px solid #86efac",
        "borderRadius": "4px",
        "color": "#166534",
    }
    hidden_btn = {
        "display": "none",
        "width": "100%",
        "padding": "6px 12px",
        "marginTop": "4px",
        "cursor": "pointer",
        "background": "#cf202f",
        "color": "white",
        "border": "0",
        "borderRadius": "6px",
        "fontSize": "12px",
        "fontWeight": "600",
    }
    return None, "", hidden_info, hidden_btn, None, None, None, None, None, None, ""


# ---------------------------------------------------------------------------
# Algorithm history visualization callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("algo-history-store", "data"),
    Output("algo-history-slider", "max"),
    Output("algo-history-slider", "value"),
    Output("algo-history-wrap", "style"),
    Input("opt-meta", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def populate_history_store(opt_meta, session_id):
    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}

    state = get_session_state(session_id)
    opt_results = state.get("opt_results")
    env = state.get("env")

    if not opt_results or not env:
        return None, 1, 0, hidden_style

    history = opt_results.get("history") or []
    algo = opt_results.get("algo", "")

    snapshot_entries = [e for e in history if "stations" in e]
    if len(snapshot_entries) < 2:
        return None, 1, 0, hidden_style

    x_scale = (env.lon_max - env.lon_min) / env.width_m
    y_scale = (env.lat_max - env.lat_min) / env.height_m

    frames = []
    for entry in snapshot_entries:
        local = np.array(entry["stations"], dtype=float)
        if local.ndim != 2 or local.shape[1] != 2:
            continue
        lon = env.lon_min + local[:, 0] * x_scale
        lat = env.lat_min + local[:, 1] * y_scale
        frames.append({
            "iter": int(entry["iter"]),
            "best_score": float(entry.get("best_score", 0)),
            "stations_geo": [[float(la), float(lo)] for la, lo in zip(lat, lon)],
        })

    if not frames:
        return None, 1, 0, hidden_style

    score_series = [
        {
            "iter": int(e["iter"]),
            "best_score": float(e.get("best_score", 0)),
            "gen_score": float(e.get("gen_best_score", e.get("current_score", e.get("best_score", 0)))),
        }
        for e in history
    ]

    algo_history_data = {"algo": algo, "frames": frames, "score_series": score_series}
    n_frames = len(frames)
    return algo_history_data, n_frames - 1, n_frames - 1, visible_style


@app.callback(
    Output("algo-history-label", "children"),
    Input("algo-history-slider", "value"),
    Input("algo-history-store", "data"),
)
def update_algo_history_label(frame_idx, algo_history):
    if not isinstance(algo_history, dict) or not algo_history.get("frames"):
        return ""
    frames = algo_history["frames"]
    n_frames = len(frames)
    idx = min(safe_int(frame_idx, 0), n_frames - 1)
    frame = frames[idx]
    algo = algo_history.get("algo", "알고리즘")
    suffix = " (최종 결과)" if idx == n_frames - 1 else ""
    return f"{algo} 수렴 과정{suffix}: {idx + 1}/{n_frames} | score: {frame['best_score']:.4f}"


@app.callback(
    Output("algo-history-chart", "figure"),
    Input("algo-history-slider", "value"),
    Input("algo-history-store", "data"),
)
def update_algo_history_chart(frame_idx, algo_history):
    if not isinstance(algo_history, dict) or not algo_history.get("score_series"):
        empty = go.Figure()
        empty.update_layout(margin={"l": 30, "r": 10, "t": 20, "b": 30}, height=150)
        return empty

    series = algo_history["score_series"]
    iters = [e["iter"] for e in series]
    best_scores = [e["best_score"] for e in series]
    gen_scores = [e.get("gen_score") for e in series]

    frames = algo_history.get("frames", [])
    n_frames = len(frames)
    idx = min(safe_int(frame_idx, 0), n_frames - 1) if n_frames > 0 else 0
    current_iter = frames[idx]["iter"] if frames else 0

    fig = go.Figure()

    if any(g is not None for g in gen_scores):
        fig.add_trace(go.Scatter(
            x=iters,
            y=gen_scores,
            mode="lines",
            name="Gen Score",
            line={"color": "#9ca3af", "width": 1, "dash": "dot"},
        ))

    fig.add_trace(go.Scatter(
        x=iters,
        y=best_scores,
        mode="lines",
        name="Best Score",
        line={"color": "#7132f5", "width": 2},
    ))

    fig.add_vline(x=current_iter, line_color="#ea580c", line_width=2, line_dash="dash")

    fig.update_layout(
        margin={"l": 30, "r": 10, "t": 10, "b": 30},
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#f7f7f7",
        xaxis={"title": "Iteration", "tickfont": {"size": 10}, "gridcolor": "#dedee5"},
        yaxis={"title": "Score", "tickfont": {"size": 10}, "gridcolor": "#dedee5"},
        height=150,
    )
    return fig


@app.callback(
    Output("sidebar-convergence-chart", "figure"),
    Output("sidebar-convergence-wrap", "style"),
    Input("algo-history-store", "data"),
    Input("opt-live-store", "data"),
)
def update_sidebar_convergence_chart(algo_history, opt_live):
    hidden = {"display": "none", "marginTop": "10px"}
    visible = {"display": "block", "marginTop": "10px"}
    empty_fig = go.Figure()
    empty_fig.update_layout(margin={"l": 30, "r": 10, "t": 10, "b": 30}, height=160)

    live_series = []
    if isinstance(opt_live, dict) and opt_live.get("running"):
        live_series = opt_live.get("score_series") or []

    if live_series:
        series = live_series
    elif isinstance(algo_history, dict) and algo_history.get("score_series"):
        series = algo_history["score_series"]
    else:
        return empty_fig, hidden

    iters = [e["iter"] for e in series]
    best_scores = [e["best_score"] for e in series]
    gen_scores = [e.get("gen_score") for e in series]

    fig = go.Figure()
    if any(g is not None for g in gen_scores):
        fig.add_trace(go.Scatter(
            x=iters, y=gen_scores, mode="lines", name="Gen",
            line={"color": "#a8acb3", "width": 1, "dash": "dot"},
        ))
    fig.add_trace(go.Scatter(
        x=iters, y=best_scores, mode="lines", name="Best",
        line={"color": "#7132f5", "width": 2},
    ))
    fig.update_layout(
        margin={"l": 30, "r": 10, "t": 10, "b": 30},
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#f7f7f7",
        xaxis={"title": "Iteration", "tickfont": {"size": 9}, "gridcolor": "#dedee5"},
        yaxis={"title": "Score", "tickfont": {"size": 9}, "gridcolor": "#dedee5"},
        height=160,
    )
    return fig, visible


@app.callback(
    Output("algo-frame-interval", "disabled"),
    Output("algo-history-slider", "value", allow_duplicate=True),
    Output("algo-play-btn", "children"),
    Input("algo-play-btn", "n_clicks"),
    State("algo-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_algo_playback(n_clicks, is_stopped):
    if not n_clicks:
        raise PreventUpdate
    if is_stopped:
        return False, 0, "⏸ 일시정지"
    return True, no_update, "▶ 재생"


@app.callback(
    Output("algo-history-slider", "value", allow_duplicate=True),
    Output("algo-frame-interval", "disabled", allow_duplicate=True),
    Output("algo-play-btn", "children", allow_duplicate=True),
    Input("algo-frame-interval", "n_intervals"),
    State("algo-history-slider", "value"),
    State("algo-history-slider", "max"),
    State("algo-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def advance_algo_frame(n_intervals, current_value, max_value, disabled):
    if disabled:
        raise PreventUpdate
    current = safe_int(current_value, 0)
    max_frame = max(0, safe_int(max_value, 0))
    if max_frame <= 0:
        raise PreventUpdate
    next_frame = current + 1
    if next_frame >= max_frame:
        return max_frame, True, "▶ 재생"
    return next_frame, False, no_update


@app.callback(
    Output("algo-history-slider", "value", allow_duplicate=True),
    Output("algo-frame-interval", "disabled", allow_duplicate=True),
    Output("algo-play-btn", "children", allow_duplicate=True),
    Input("algo-reset-btn", "n_clicks"),
    State("algo-history-slider", "max"),
    prevent_initial_call=True,
)
def reset_algo_frame(n_clicks, max_value):
    if not n_clicks:
        raise PreventUpdate
    return safe_int(max_value, 0), True, "▶ 재생"


# ===========================================================================
# Sweep 콜백
# ===========================================================================

_SWEEP_INPUT_STYLE = {
    "width": "100%", "padding": "3px 5px",
    "borderRadius": "4px", "border": "1px solid #d1d5db", "fontSize": "12px",
}


@app.callback(
    Output("sweep-params-container", "children"),
    Output("sweep-algo-display", "children"),
    Input("algo-select", "value"),
)
def render_sweep_params_ui(algo):
    display = f"알고리즘: {algo}" if algo else ""
    optimizer = get_optimizer(algo)

    def _param_row(name, label, default_min, default_max, default_steps=5):
        return html.Div([
            html.Div(
                [
                    dcc.Checklist(
                        id={"type": "sweep-p-enabled", "name": name},
                        options=[{"label": "", "value": "on"}],
                        value=[],
                        style={"display": "inline-block", "marginRight": "6px"},
                        inputStyle={"cursor": "pointer"},
                    ),
                    html.Span(label, style={"fontSize": "12px", "fontWeight": "600",
                                            "verticalAlign": "middle"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("min", style={"fontSize": "10px", "color": "#6b7280",
                                                   "display": "block", "marginBottom": "1px"}),
                         dcc.Input(id={"type": "sweep-p-min", "name": name},
                                   type="number", value=default_min, style=_SWEEP_INPUT_STYLE)],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [html.Label("max", style={"fontSize": "10px", "color": "#6b7280",
                                                   "display": "block", "marginBottom": "1px"}),
                         dcc.Input(id={"type": "sweep-p-max", "name": name},
                                   type="number", value=default_max, style=_SWEEP_INPUT_STYLE)],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [html.Label("단계", style={"fontSize": "10px", "color": "#6b7280",
                                                    "display": "block", "marginBottom": "1px"}),
                         dcc.Input(id={"type": "sweep-p-steps", "name": name},
                                   type="number", value=default_steps, min=2, max=20, step=1,
                                   style=_SWEEP_INPUT_STYLE)],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "4px", "marginBottom": "8px", "paddingLeft": "20px"},
            ),
        ])

    # 기지국 수는 항상 첫 번째 행으로 표시
    rows = [
        _param_row("__k__", "기지국 수 (k)", default_min=1, default_max=10, default_steps=5),
        html.Hr(style={"border": "none", "borderTop": "1px solid #e5e7eb", "margin": "4px 0 8px"}),
    ]

    if optimizer:
        for p in optimizer.hyperparams:
            if p.kind not in ("int", "float"):
                continue
            rows.append(_param_row(p.name, p.label or p.name, p.min, p.max))

    return rows, display


@app.callback(
    Output("sweep-run-btn", "disabled"),
    Output("sweep-poll-interval", "disabled"),
    Output("sweep-status", "children"),
    Input("sweep-run-btn", "n_clicks"),
    State("session-id", "data"),
    State("algo-select", "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "id"),
    State("n-stations", "value"),
    State({"type": "sweep-p-enabled", "name": ALL}, "value"),
    State({"type": "sweep-p-enabled", "name": ALL}, "id"),
    State({"type": "sweep-p-min",     "name": ALL}, "value"),
    State({"type": "sweep-p-max",     "name": ALL}, "value"),
    State({"type": "sweep-p-steps",   "name": ALL}, "value"),
    State("ui-tx-power", "value"),
    State("ui-path-loss-exp", "value"),
    State("ui-bandwidth-mhz", "value"),
    State("ui-sinr-threshold", "value"),
    State("ui-max-coord", "value"),
    State("spec-mode", "value"),
    State("station-specs-store", "data"),
    State("score-mode", "value"),
    State("spectral-eff-mode", "value"),
    State("operation-policy", "value"),
    prevent_initial_call=True,
)
def start_sweep_job(
    n_clicks, session_id, algo,
    hp_values, hp_ids,
    n_stations,
    enabled_values, enabled_ids, min_values, max_values, steps_values,
    ui_tx_power, ui_path_loss_exp, ui_bandwidth_mhz, ui_sinr_threshold, ui_max_coord,
    spec_mode, station_specs,
    score_mode, spectral_eff_mode, operation_policy,
):
    if not n_clicks:
        raise PreventUpdate

    def _err(msg):
        return False, True, html.Span(msg, style={"color": "#cf202f", "fontWeight": "600"})

    state = get_session_state(session_id)

    if state.get("env") is None:
        return _err("먼저 환경 데이터를 생성해주세요.")
    if state.get("sweep_progress", {}).get("running"):
        return False, False, html.Span("이미 Sweep 실행 중입니다.", style={"color": "#b45309"})

    optimizer = get_optimizer(algo)
    hp_defaults = {p.name: p.default for p in optimizer.hyperparams}

    # 활성화된 파라미터별 sweep 값 수집
    sweep_params = []
    for enabled_val, id_obj, v_min, v_max, n_steps in zip(
        enabled_values, enabled_ids, min_values, max_values, steps_values
    ):
        if "on" not in (enabled_val or []):
            continue
        name = id_obj["name"]
        # __k__ 는 hyperparam 목록 밖의 특수 변수
        kind = "int" if name == "__k__" else next(
            (p.kind for p in optimizer.hyperparams if p.name == name), "float"
        )
        fmin = safe_float(v_min, None)
        fmax = safe_float(v_max, None)
        if fmin is None or fmax is None or fmin >= fmax:
            return _err(f"파라미터 '{name}'의 유효한 최솟값/최댓값을 입력해주세요 (min < max).")
        steps = max(2, safe_int(n_steps, 5))
        raw = np.linspace(fmin, fmax, steps)
        vals = sorted(set(int(round(v)) for v in raw)) if kind == "int" else [float(v) for v in raw]
        sweep_params.append({"name": name, "kind": kind, "values": vals})

    if not sweep_params:
        return _err("Sweep할 파라미터를 하나 이상 체크해주세요.")

    total_combos = 1
    for p in sweep_params:
        total_combos *= len(p["values"])
    if total_combos > 500:
        return _err(f"조합 수 {total_combos}이 너무 많습니다 (최대 500). 단계 수를 줄여주세요.")

    sweep_param_names = {p["name"] for p in sweep_params}
    all_hyperparams = _parse_hyperparams(hp_values, hp_ids, hp_defaults)
    # __k__ 는 hyperparam이 아니므로 fixed_hyperparams 에서도 제외
    fixed_hyperparams = {k: v for k, v in all_hyperparams.items()
                         if k not in sweep_param_names and k != "__k__"}

    prop = prop_params_base(
        path_loss_exponent=safe_float(ui_path_loss_exp, 3.5),
        bandwidth_mhz=safe_float(ui_bandwidth_mhz, 10.0),
        sinr_threshold_db=safe_float(ui_sinr_threshold, 3.0),
        max_coord_stations=safe_int(ui_max_coord, 1),
    )

    state["sweep_config"] = {
        "algo": algo,
        "sweep_params": sweep_params,
        "fixed_hyperparams": fixed_hyperparams,
        "k": safe_int(n_stations, 5),
        "prop": prop,
        "spec_mode": spec_mode,
        "station_specs": station_specs,
        "ui_tx_power": ui_tx_power,
        "score_mode": score_mode or "traffic",
        "spectral_efficiency_mode": spectral_eff_mode or "shannon",
        "weight_scale": 1.0,
        "operation_policy": normalize_operation_policy(operation_policy),
    }
    state["sweep_progress"] = {
        "running": True, "done": False, "error": None,
        "current": 0, "total": total_combos,
    }

    threading.Thread(target=_run_sweep_thread, args=(session_id,), daemon=True).start()

    param_names_str = ", ".join(p["name"] for p in sweep_params)
    return True, False, html.Span(
        f"Sweep 시작: [{param_names_str}] {total_combos}개 조합",
        style={"color": "#7132f5"},
    )


def _run_sweep_thread(session_id: str) -> None:
    import itertools as _itertools
    try:
        state = get_session_state(session_id)
        cfg = state.get("sweep_config")
        env = state.get("env")
        if cfg is None or env is None:
            state["sweep_progress"] = {"running": False, "done": False,
                                        "error": "설정 또는 환경이 없습니다."}
            return

        algo = cfg["algo"]
        sweep_params = cfg["sweep_params"]        # [{"name", "kind", "values"}, ...]
        fixed_hyperparams = cfg["fixed_hyperparams"]
        k = cfg["k"]
        prop = cfg["prop"]
        score_mode = cfg.get("score_mode", "traffic")
        spectral_efficiency_mode = cfg.get("spectral_efficiency_mode", "shannon")
        weight_scale = float(cfg.get("weight_scale", 1.0))
        operation_policy = normalize_operation_policy(cfg.get("operation_policy"))

        optimizer = get_optimizer(algo)
        k_is_swept = any(p["name"] == "__k__" for p in sweep_params)

        def _build_problem(k_val):
            tx = tx_power_for_k(
                k_val,
                safe_float(cfg["ui_tx_power"], 43.0),
                cfg["spec_mode"],
                cfg["station_specs"],
            )
            r = radius_from_tx(tx, prop)
            prob = ProblemInput.from_env(
                env, radius_m=r, capacity=np.full(k_val, 1e10),
                station_candidate_points=env.station_candidate_points,
                path_loss_exponent=prop["path_loss_exponent"],
                path_loss_ref_db=prop["path_loss_ref_db"],
                tx_power_dbm=tx,
                bandwidth_mhz=prop["bandwidth_mhz"],
                sinr_threshold_db=prop["sinr_threshold_db"],
                noise_floor_dbm=prop["noise_floor_dbm"],
                score_mode=score_mode,
                spectral_efficiency_mode=spectral_efficiency_mode,
                weight_scale=weight_scale,
                interference_threshold_dbm=prop["noise_floor_dbm"],
                max_coord_stations=prop.get("max_coord_stations", 1),
            )
            return prob, tx

        if not k_is_swept:
            problem, tx_k = _build_problem(k)
        else:
            problem, tx_k = None, None

        combos = list(_itertools.product(*[p["values"] for p in sweep_params]))
        sweep_results = []
        _last_k = None
        for i, combo in enumerate(combos):
            param_combo = {p["name"]: val for p, val in zip(sweep_params, combo)}

            k_val = int(param_combo.pop("__k__", k))
            if k_is_swept and k_val != _last_k:
                problem, tx_k = _build_problem(k_val)
                _last_k = k_val

            hyperparams = {**fixed_hyperparams, **param_combo}
            result = optimizer.optimize(problem, n_stations=k_val, **hyperparams)
            metrics = dict(result.metrics)
            stations_geo = convert_to_geo(result.stations, problem)
            stations_df = pd.DataFrame(stations_geo, columns=["lat", "lon"])

            display_combo = {"k": k_val, **param_combo} if k_is_swept else param_combo

            sweep_results.append({
                "param_combo": display_combo,
                "score": float(result.score),
                "covered_traffic": float(metrics.get("covered_traffic", 0)),
                "covered_area": int(metrics.get("covered_area", 0)),
                "opt_results": {
                    "algo": algo,
                    "score": float(result.score),
                    "stations_geo": stations_df.to_dict("records"),
                    "history": result.history,
                    "prop_params": {**prop, "tx_power_dbm": tx_k.tolist()},
                    "operation_policy": operation_policy,
                },
                "opt_stats": metrics,
            })
            state["sweep_progress"] = {
                "running": True, "done": False, "error": None,
                "current": i + 1, "total": len(combos),
            }

        state["sweep_results"] = sweep_results
        state["sweep_progress"] = {"running": False, "done": True, "error": None,
                                    "current": len(combos), "total": len(combos)}
    except Exception:
        import traceback
        tb = traceback.format_exc()
        _opt_logger.error("sweep thread error: %s", tb)
        try:
            state["sweep_progress"] = {"running": False, "done": False, "error": tb}
        except Exception:
            pass


@app.callback(
    Output("sweep-status", "children", allow_duplicate=True),
    Output("sweep-meta", "data"),
    Output("sweep-run-btn", "disabled", allow_duplicate=True),
    Output("sweep-poll-interval", "disabled", allow_duplicate=True),
    Input("sweep-poll-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_sweep_progress(n_intervals, session_id):
    state = get_session_state(session_id)
    progress = state.get("sweep_progress")

    if not progress:
        raise PreventUpdate

    if progress.get("error"):
        msg = html.Span(f"오류: {str(progress['error'])[:120]}",
                        style={"color": "#cf202f", "fontSize": "11px"})
        return msg, no_update, False, True

    if progress.get("done"):
        cur = progress.get("current", 0)
        tot = progress.get("total", 0)
        msg = html.Span(f"완료: {cur}/{tot}",
                        style={"color": "#026b3f", "fontWeight": "600"})
        return msg, version_token(), False, True

    cur = progress.get("current", 0)
    tot = progress.get("total", 1)
    pct = int(cur / max(tot, 1) * 100)
    msg = html.Div([
        html.Span(f"{cur} / {tot} 완료 ({pct}%)",
                  style={"fontSize": "12px", "color": "#7132f5"}),
        html.Div(style={
            "height": "4px", "background": "#dedee5", "borderRadius": "2px",
            "marginTop": "4px",
        }, children=[
            html.Div(style={
                "height": "100%", "width": f"{pct}%",
                "background": "#7132f5", "borderRadius": "2px",
                "transition": "width 0.3s ease",
            })
        ]),
    ])
    return msg, no_update, no_update, no_update


@app.callback(
    Output("sweep-result-chart", "figure"),
    Output("sweep-result-table", "children"),
    Input("sweep-meta", "data"),
    State("session-id", "data"),
)
def render_sweep_results(sweep_meta, session_id):
    if not sweep_meta:
        return go.Figure(), []

    state = get_session_state(session_id)
    results = state.get("sweep_results")
    if not results:
        return go.Figure(), []

    param_names = list(results[0]["param_combo"].keys())
    n_params = len(param_names)

    # DataFrame: 파라미터 컬럼 + score + covered_traffic
    rows = []
    for r in results:
        row = {**r["param_combo"], "score": r["score"], "covered_traffic": r["covered_traffic"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    best_idx = int(df["score"].idxmax())

    # ── 차트 분기 ──────────────────────────────────────────────
    fig = go.Figure()
    base_layout = dict(
        margin={"l": 35, "r": 10, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        autosize=True,
    )

    if n_params == 1:
        name0 = param_names[0]
        sizes = [14 if i == best_idx else 7 for i in range(len(df))]
        colors = ["#149e61" if i == best_idx else "#7132f5" for i in range(len(df))]
        fig.add_trace(go.Scatter(
            x=df[name0], y=df["score"],
            mode="lines+markers",
            marker={"size": sizes, "color": colors},
            line={"color": "#a8b8cc", "width": 1.5},
        ))
        fig.update_layout(
            xaxis_title=name0, yaxis_title="Score",
            xaxis={"gridcolor": "#dedee5"}, yaxis={"gridcolor": "#dedee5"},
            **base_layout,
        )

    elif n_params == 2:
        name0, name1 = param_names
        pivot = df.pivot_table(index=name1, columns=name0, values="score", aggfunc="max")
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale="Blues",
            colorbar={"thickness": 10, "title": "Score"},
        ))
        # 최고점 마커
        best_row = df.iloc[best_idx]
        fig.add_trace(go.Scatter(
            x=[str(best_row[name0])], y=[str(best_row[name1])],
            mode="markers",
            marker={"symbol": "star", "size": 14, "color": "#149e61"},
            name="Best",
            showlegend=False,
        ))
        fig.update_layout(
            xaxis_title=name0, yaxis_title=name1,
            **base_layout,
        )

    else:  # 3개 이상 → parallel coordinates
        dims = [
            {"label": n, "values": df[n].tolist()}
            for n in param_names
        ]
        dims.append({"label": "score", "values": df["score"].tolist()})
        fig.add_trace(go.Parcoords(
            line={"color": df["score"], "colorscale": "Blues", "showscale": True},
            dimensions=dims,
        ))
        fig.update_layout(**base_layout)

    # ── 테이블 ──────────────────────────────────────────────────
    cols = (
        [{"name": n, "id": n} for n in param_names]
        + [{"name": "score", "id": "score"}, {"name": "traffic", "id": "covered_traffic"}]
    )
    table = dash_table.DataTable(
        id="sweep-datatable",
        data=df.round(4).to_dict("records"),
        columns=cols,
        page_size=8,
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto", "maxHeight": "160px", "overflowY": "auto"},
        style_cell={"fontSize": "11px", "padding": "3px 6px", "cursor": "pointer"},
        style_header={"fontSize": "11px", "fontWeight": "700"},
        style_data_conditional=[
            {
                "if": {"row_index": best_idx},
                "backgroundColor": "rgba(20, 158, 97, 0.10)",
                "fontWeight": "700",
                "color": "#026b3f",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "rgba(133, 91, 251, 0.10)",
                "border": "1px solid #7132f5",
            },
        ],
    )
    hint = html.Div(
        "행을 클릭하면 해당 결과를 지도에 표시합니다.",
        style={"fontSize": "11px", "color": "#6b7280", "marginTop": "4px"},
    )

    return fig, [table, hint]


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("sweep-status", "children", allow_duplicate=True),
    Input("sweep-apply-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_sweep_best(n_clicks, session_id):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)
    results = state.get("sweep_results")
    if not results:
        return no_update, html.Span("Sweep 결과가 없습니다.",
                                     style={"color": "#cf202f", "fontSize": "12px"})

    best = max(results, key=lambda r: r["score"])
    state["opt_results"] = best["opt_results"]
    state["opt_stats"] = best["opt_stats"]
    state.pop("operation_results", None)

    combo_str = ", ".join(f"{k}={v:.3g}" for k, v in best["param_combo"].items())
    msg = html.Span(
        f"적용 완료: {combo_str}, score={best['score']:.1f}",
        style={"color": "#026b3f", "fontWeight": "600", "fontSize": "12px"},
    )
    return version_token(), msg


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("sweep-status", "children", allow_duplicate=True),
    Input("sweep-datatable", "selected_rows"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_sweep_row(selected_rows, session_id):
    if not selected_rows:
        raise PreventUpdate

    idx = selected_rows[0]
    state = get_session_state(session_id)
    results = state.get("sweep_results")
    if not results or idx >= len(results):
        raise PreventUpdate

    chosen = results[idx]
    state["opt_results"] = chosen["opt_results"]
    state["opt_stats"] = chosen["opt_stats"]
    state.pop("operation_results", None)

    combo_str = ", ".join(f"{k}={v:.3g}" for k, v in chosen["param_combo"].items())
    msg = html.Span(
        f"적용: {combo_str} | score={chosen['score']:.1f}",
        style={"color": "#7132f5", "fontWeight": "600", "fontSize": "12px"},
    )
    return version_token(), msg


# ── Sweep 모드 전환 ───────────────────────────────────────────────────────

@app.callback(
    Output("sweep-mode1-panel", "style"),
    Output("sweep-mode2-panel", "style"),
    Input("sweep-mode", "value"),
)
def toggle_sweep_mode_panels(mode):
    if mode == "mode1":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


# ── 알고리즘 비교 ─────────────────────────────────────────────────────────

@app.callback(
    Output("algo-compare-run-btn", "disabled"),
    Output("algo-compare-poll-interval", "disabled"),
    Output("algo-compare-status", "children"),
    Input("algo-compare-run-btn", "n_clicks"),
    State("session-id", "data"),
    State({"type": "mode2-algo-check", "algo": ALL}, "value"),
    State({"type": "mode2-algo-check", "algo": ALL}, "id"),
    State({"type": "mode2-hp", "algo": ALL, "param": ALL}, "value"),
    State({"type": "mode2-hp", "algo": ALL, "param": ALL}, "id"),
    State("n-stations", "value"),
    State("ui-tx-power", "value"),
    State("ui-path-loss-exp", "value"),
    State("ui-bandwidth-mhz", "value"),
    State("ui-sinr-threshold", "value"),
    State("ui-max-coord", "value"),
    State("spec-mode", "value"),
    State("station-specs-store", "data"),
    State("score-mode", "value"),
    State("spectral-eff-mode", "value"),
    State("operation-policy", "value"),
    prevent_initial_call=True,
)
def start_algo_compare_job(
    n_clicks, session_id,
    check_values, check_ids,
    hp_values, hp_ids,
    n_stations,
    ui_tx_power, ui_path_loss_exp, ui_bandwidth_mhz, ui_sinr_threshold, ui_max_coord,
    spec_mode, station_specs,
    score_mode, spectral_eff_mode, operation_policy,
):
    if not n_clicks:
        raise PreventUpdate

    def _err(msg):
        return False, True, html.Span(msg, style={"color": "#cf202f", "fontWeight": "600"})

    state = get_session_state(session_id)

    # 체크된 알고리즘 목록 도출
    selected_algos = [
        id_obj["algo"]
        for val, id_obj in zip(check_values, check_ids)
        if val and "on" in val
    ]

    if state.get("env") is None:
        return _err("먼저 환경 데이터를 생성해주세요.")
    if not selected_algos:
        return _err("비교할 알고리즘을 하나 이상 선택해주세요.")
    if state.get("algo_compare_progress", {}).get("running"):
        return False, False, html.Span("이미 비교 실행 중입니다.", style={"color": "#b45309"})

    # 알고리즘별 HP 수집: {algo_name: {param_name: value}}
    algo_hyperparams: dict[str, dict] = {}
    for val, id_obj in zip(hp_values, hp_ids):
        algo = id_obj["algo"]
        param = id_obj["param"]
        if algo not in selected_algos:
            continue
        optimizer = get_optimizer(algo)
        hp_def = next((h for h in optimizer.hyperparams if h.name == param), None)
        if hp_def is None:
            continue
        if hp_def.kind == "bool":
            parsed = "on" in (val or [])
        elif hp_def.kind == "int":
            parsed = safe_int(val, int(hp_def.default))
        else:
            parsed = safe_float(val, float(hp_def.default))
        algo_hyperparams.setdefault(algo, {})[param] = parsed

    # 누락된 알고리즘은 기본값으로 채움
    for algo in selected_algos:
        if algo not in algo_hyperparams:
            optimizer = get_optimizer(algo)
            algo_hyperparams[algo] = {h.name: h.default for h in optimizer.hyperparams}

    prop = prop_params_base(
        path_loss_exponent=safe_float(ui_path_loss_exp, 3.5),
        bandwidth_mhz=safe_float(ui_bandwidth_mhz, 10.0),
        sinr_threshold_db=safe_float(ui_sinr_threshold, 3.0),
        max_coord_stations=safe_int(ui_max_coord, 1),
    )

    state["algo_compare_config"] = {
        "selected_algos": selected_algos,
        "algo_hyperparams": algo_hyperparams,
        "k": safe_int(n_stations, 5),
        "prop": prop,
        "spec_mode": spec_mode,
        "station_specs": station_specs,
        "ui_tx_power": ui_tx_power,
        "score_mode": score_mode or "traffic",
        "spectral_efficiency_mode": spectral_eff_mode or "shannon",
        "weight_scale": 1.0,
        "operation_policy": normalize_operation_policy(operation_policy),
    }
    state["algo_compare_progress"] = {
        "running": True, "done": False, "error": None,
        "current": 0, "total": len(selected_algos), "current_algo": "",
    }

    threading.Thread(target=_run_algo_compare_thread, args=(session_id,), daemon=True).start()

    return True, False, html.Span(
        f"비교 시작: {len(selected_algos)}개 알고리즘",
        style={"color": "#7132f5"},
    )


def _run_algo_compare_thread(session_id: str) -> None:
    import time as _time
    try:
        state = get_session_state(session_id)
        cfg = state.get("algo_compare_config")
        env = state.get("env")
        if cfg is None or env is None:
            state["algo_compare_progress"] = {
                "running": False, "done": False,
                "error": "설정 또는 환경이 없습니다.",
            }
            return

        selected_algos = cfg["selected_algos"]
        k = cfg["k"]
        prop = cfg["prop"]
        score_mode = cfg.get("score_mode", "traffic")
        spectral_efficiency_mode = cfg.get("spectral_efficiency_mode", "shannon")
        weight_scale = float(cfg.get("weight_scale", 1.0))
        operation_policy = normalize_operation_policy(cfg.get("operation_policy"))

        def _build_problem(k_val):
            tx = tx_power_for_k(
                k_val,
                safe_float(cfg["ui_tx_power"], 43.0),
                cfg["spec_mode"],
                cfg["station_specs"],
            )
            r = radius_from_tx(tx, prop)
            prob = ProblemInput.from_env(
                env, radius_m=r, capacity=np.full(k_val, 1e10),
                station_candidate_points=env.station_candidate_points,
                path_loss_exponent=prop["path_loss_exponent"],
                path_loss_ref_db=prop["path_loss_ref_db"],
                tx_power_dbm=tx,
                bandwidth_mhz=prop["bandwidth_mhz"],
                sinr_threshold_db=prop["sinr_threshold_db"],
                noise_floor_dbm=prop["noise_floor_dbm"],
                score_mode=score_mode,
                spectral_efficiency_mode=spectral_efficiency_mode,
                weight_scale=weight_scale,
                interference_threshold_dbm=prop["noise_floor_dbm"],
                max_coord_stations=prop.get("max_coord_stations", 1),
            )
            return prob, tx

        problem, tx_k = _build_problem(k)

        algo_hyperparams = cfg.get("algo_hyperparams", {})

        comparison_results = []
        for i, algo_name in enumerate(selected_algos):
            state["algo_compare_progress"]["current"] = i
            state["algo_compare_progress"]["current_algo"] = algo_name

            optimizer = get_optimizer(algo_name)
            hyperparams = algo_hyperparams.get(
                algo_name, {p.name: p.default for p in optimizer.hyperparams}
            )

            t0 = _time.time()
            result = optimizer.optimize(problem, n_stations=k, **hyperparams)
            elapsed = _time.time() - t0

            metrics = dict(result.metrics)
            total_t = float(metrics.get("total_traffic", 1) or 1)
            total_a = int(metrics.get("total_area", 1) or 1)
            stations_geo = convert_to_geo(result.stations, problem)
            stations_df = pd.DataFrame(stations_geo, columns=["lat", "lon"])

            comparison_results.append({
                "algo": algo_name,
                "score": float(result.score),
                "covered_traffic": float(metrics.get("covered_traffic", 0)),
                "coverage_pct": float(metrics.get("covered_traffic", 0)) / total_t * 100,
                "area_pct": int(metrics.get("covered_area", 0)) / total_a * 100,
                "mean_sinr_db": metrics.get("mean_sinr_db"),
                "total_throughput_mbps": float(metrics.get("total_throughput_mbps", 0)),
                "elapsed_sec": elapsed,
                "opt_results": {
                    "algo": algo_name,
                    "score": float(result.score),
                    "stations_geo": stations_df.to_dict("records"),
                    "history": result.history,
                    "prop_params": {**prop, "tx_power_dbm": tx_k.tolist()},
                    "score_mode": score_mode,
                    "spectral_efficiency_mode": spectral_efficiency_mode,
                    "weight_scale": weight_scale,
                    "operation_policy": operation_policy,
                },
                "opt_stats": metrics,
            })

        state["algo_compare_results"] = comparison_results
        state["algo_compare_progress"] = {
            "running": False, "done": True, "error": None,
            "current": len(selected_algos), "total": len(selected_algos),
            "current_algo": "",
        }
    except Exception:
        import traceback
        tb = traceback.format_exc()
        _opt_logger.error("algo_compare thread error: %s", tb)
        try:
            state["algo_compare_progress"] = {"running": False, "done": False, "error": tb}
        except Exception:
            pass


@app.callback(
    Output("algo-compare-status", "children", allow_duplicate=True),
    Output("algo-compare-meta", "data"),
    Output("algo-compare-run-btn", "disabled", allow_duplicate=True),
    Output("algo-compare-poll-interval", "disabled", allow_duplicate=True),
    Input("algo-compare-poll-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_algo_compare_progress(n_intervals, session_id):
    state = get_session_state(session_id)
    progress = state.get("algo_compare_progress")

    if not progress:
        raise PreventUpdate

    if progress.get("error"):
        msg = html.Span(f"오류: {str(progress['error'])[:120]}",
                        style={"color": "#cf202f", "fontSize": "11px"})
        return msg, no_update, False, True

    if progress.get("done"):
        cur = progress.get("current", 0)
        tot = progress.get("total", 0)
        msg = html.Span(f"완료: {cur}/{tot}개 알고리즘",
                        style={"color": "#026b3f", "fontWeight": "600"})
        return msg, version_token(), False, True

    cur = progress.get("current", 0)
    tot = progress.get("total", 1)
    algo_name = progress.get("current_algo", "")
    pct = int(cur / max(tot, 1) * 100)
    msg = html.Div([
        html.Span(f"{cur}/{tot} 완료 ({pct}%) — {algo_name}",
                  style={"fontSize": "12px", "color": "#7132f5"}),
        html.Div(style={
            "height": "4px", "background": "#dedee5", "borderRadius": "2px",
            "marginTop": "4px",
        }, children=[
            html.Div(style={
                "height": "100%", "width": f"{pct}%",
                "background": "#7132f5", "borderRadius": "2px",
                "transition": "width 0.3s ease",
            })
        ]),
    ])
    return msg, no_update, no_update, no_update


@app.callback(
    Output("algo-compare-results", "children"),
    Input("algo-compare-meta", "data"),
    State("session-id", "data"),
)
def render_algo_compare_results(compare_meta, session_id):
    if not compare_meta:
        return []

    state = get_session_state(session_id)
    results = state.get("algo_compare_results")
    if not results:
        return []

    algo_names = [r["algo"] for r in results]
    scores = [r["score"] for r in results]
    throughputs = [r["total_throughput_mbps"] for r in results]
    best_idx = int(scores.index(max(scores)))

    # 가로 막대 차트: score 기준 정렬, throughput 색상
    sort_order = sorted(range(len(scores)), key=lambda i: scores[i])
    sorted_names = [algo_names[i] for i in sort_order]
    sorted_scores = [scores[i] for i in sort_order]
    sorted_tp = [throughputs[i] for i in sort_order]
    bar_colors = [
        "#149e61" if algo_names[i] == algo_names[best_idx] else "#7132f5"
        for i in sort_order
    ]

    fig = go.Figure(go.Bar(
        x=sorted_scores,
        y=sorted_names,
        orientation="h",
        marker={"color": bar_colors},
        text=[f"{s:.1f}" for s in sorted_scores],
        textposition="outside",
        customdata=sorted_tp,
        hovertemplate="%{y}<br>Score: %{x:.2f}<br>처리량: %{customdata:.1f} Mbps<extra></extra>",
    ))
    fig.update_layout(
        margin={"l": 10, "r": 50, "t": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "Score", "gridcolor": "#dedee5"},
        yaxis={"gridcolor": "#dedee5"},
        showlegend=False,
        height=max(160, len(results) * 40 + 60),
    )

    # 비교 테이블
    table_rows = []
    for r in results:
        sinr_str = f"{r['mean_sinr_db']:.1f}" if r["mean_sinr_db"] is not None else "-"
        table_rows.append({
            "algo": r["algo"],
            "score": round(r["score"], 2),
            "coverage_pct": round(r["coverage_pct"], 1),
            "throughput_mbps": round(r["total_throughput_mbps"], 1),
            "area_pct": round(r["area_pct"], 1),
            "sinr_db": sinr_str,
            "elapsed_sec": round(r["elapsed_sec"], 2),
        })
    table_best_idx = int(max(range(len(table_rows)), key=lambda i: table_rows[i]["score"]))

    table = dash_table.DataTable(
        id="algo-compare-datatable",
        data=table_rows,
        columns=[
            {"name": "알고리즘", "id": "algo"},
            {"name": "Score", "id": "score"},
            {"name": "커버리지(%)", "id": "coverage_pct"},
            {"name": "처리량(Mbps)", "id": "throughput_mbps"},
            {"name": "면적커버(%)", "id": "area_pct"},
            {"name": "SINR(dB)", "id": "sinr_db"},
            {"name": "시간(s)", "id": "elapsed_sec"},
        ],
        page_size=10,
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "11px", "padding": "3px 6px", "cursor": "pointer"},
        style_header={"fontSize": "11px", "fontWeight": "700"},
        style_data_conditional=[
            {
                "if": {"row_index": table_best_idx},
                "backgroundColor": "rgba(20, 158, 97, 0.10)",
                "fontWeight": "700",
                "color": "#026b3f",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "rgba(133, 91, 251, 0.10)",
                "border": "1px solid #7132f5",
            },
        ],
    )
    hint = html.Div(
        "행을 클릭하면 해당 알고리즘 결과를 지도에 표시합니다.",
        style={"fontSize": "11px", "color": "#6b7280", "marginTop": "4px"},
    )

    apply_btn = html.Button(
        "최적 알고리즘 결과 적용",
        id="algo-compare-apply-btn",
        n_clicks=0,
        className="primary-button",
    )

    return [
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"marginTop": "8px"}),
        html.Div([table, hint], style={"marginTop": "6px"}),
        apply_btn,
    ]


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("algo-compare-status", "children", allow_duplicate=True),
    Input("algo-compare-datatable", "selected_rows"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_algo_compare_row(selected_rows, session_id):
    if not selected_rows:
        raise PreventUpdate

    idx = selected_rows[0]
    state = get_session_state(session_id)
    results = state.get("algo_compare_results")
    if not results or idx >= len(results):
        raise PreventUpdate

    chosen = results[idx]
    state["opt_results"] = chosen["opt_results"]
    state["opt_stats"] = chosen["opt_stats"]
    state.pop("operation_results", None)

    msg = html.Span(
        f"적용: {chosen['algo']} | score={chosen['score']:.2f}",
        style={"color": "#7132f5", "fontWeight": "600", "fontSize": "12px"},
    )
    return version_token(), msg


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("algo-compare-status", "children", allow_duplicate=True),
    Input("algo-compare-apply-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_algo_compare_best(n_clicks, session_id):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)
    results = state.get("algo_compare_results")
    if not results:
        return no_update, html.Span("비교 결과가 없습니다.",
                                    style={"color": "#cf202f", "fontSize": "12px"})

    best = max(results, key=lambda r: r["score"])
    state["opt_results"] = best["opt_results"]
    state["opt_stats"] = best["opt_stats"]
    state.pop("operation_results", None)

    msg = html.Span(
        f"적용 완료: {best['algo']}, score={best['score']:.2f}",
        style={"color": "#026b3f", "fontWeight": "600", "fontSize": "12px"},
    )
    return version_token(), msg


# ── 사이드바 토글 콜백 ─────────────────────────────────────────────────────

@app.callback(
    Output("left-sidebar", "style"),
    Output("left-sidebar-open", "data"),
    Output("left-toggle-btn", "children"),
    Output("left-sidebar-body", "style"),
    Input("left-toggle-btn", "n_clicks"),
    State("left-sidebar-open", "data"),
    prevent_initial_call=True,
)
def toggle_left_sidebar(n_clicks, is_open):
    _base = {
        "height": "100vh", "display": "flex", "flexDirection": "column",
        "background": "#ffffff", "borderRight": "1px solid #dedee5",
        "boxSizing": "border-box", "padding": "0",
        "transition": "width 0.2s ease, min-width 0.2s ease",
    }
    if is_open:
        return ({**_base, "width": "44px", "minWidth": "44px", "overflow": "hidden"},
                False, "►", {"display": "none"})
    else:
        return ({**_base, "width": "320px", "minWidth": "320px"},
                True, "◄", {"overflowY": "auto", "flex": "1"})


@app.callback(
    Output("right-sidebar", "style"),
    Output("right-sidebar-open", "data"),
    Output("right-toggle-btn", "children"),
    Output("right-sidebar-body", "style"),
    Input("right-toggle-btn", "n_clicks"),
    State("right-sidebar-open", "data"),
    prevent_initial_call=True,
)
def toggle_right_sidebar(n_clicks, is_open):
    _base = {
        "height": "100vh", "display": "flex", "flexDirection": "column",
        "borderLeft": "1px solid #dedee5", "background": "#ffffff",
        "boxSizing": "border-box",
        "transition": "width 0.2s ease, min-width 0.2s ease",
    }
    if is_open:
        return ({**_base, "width": "44px", "minWidth": "44px", "overflow": "hidden"},
                False, "◄", {"display": "none"})
    else:
        return ({**_base, "width": "420px", "minWidth": "420px"},
                True, "►", {"overflowY": "auto", "flex": "1", "padding": "16px"})


# ── 영역 미지정 상태에서 생성 버튼으로 사각형 그리기 활성화 ────────────────

app.clientside_callback(
    """
    function(n, customRegion, isDrawing) {
        if (!n) {
            return window.dash_clientside.no_update;
        }
        var hasRegion = customRegion && customRegion.width_km && customRegion.height_km;
        if (hasRegion) {
            return window.dash_clientside.no_update;
        }
        if (isDrawing) {
            window.setTimeout(function() {
                var cancelAction = document.querySelector(".leaflet-draw-actions a");
                if (cancelAction) {
                    cancelAction.click();
                    return;
                }
                document.dispatchEvent(new KeyboardEvent("keydown", {
                    key: "Escape",
                    code: "Escape",
                    keyCode: 27,
                    which: 27,
                    bubbles: true
                }));
            }, 0);
            return false;
        }
        window.setTimeout(function() {
            var drawButton = document.querySelector(".leaflet-draw-draw-rectangle");
            if (drawButton) {
                drawButton.click();
            }
        }, 0);
        return true;
    }
    """,
    Output("region-draw-activate-dummy", "data"),
    Input("create-env-btn", "n_clicks"),
    State("custom-region-store", "data"),
    State("region-draw-activate-dummy", "data"),
    prevent_initial_call=True,
)


# ── 가상 데이터 생성 버튼 즉시 비활성화 ────────────────────────────────────

app.clientside_callback(
    "function(n) { return true; }",
    Output("create-env-btn", "disabled", allow_duplicate=True),
    Input("create-env-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ── 드래그 리사이즈 (클라이언트사이드) ─────────────────────────────────────

app.clientside_callback(
    """
    function(leftHandleId, rightHandleId) {
        function initResize(handleId, sidebarId, side) {
            var handle = document.getElementById(handleId);
            if (!handle || handle._resizeInit) return;
            handle._resizeInit = true;
            handle.addEventListener('mousedown', function(e) {
                e.preventDefault();
                var sidebar = document.getElementById(sidebarId);
                if (!sidebar) return;
                var startX = e.clientX;
                var startW = sidebar.getBoundingClientRect().width;
                var minW = 32, maxW = 600;
                function onMove(e) {
                    var delta = (side === 'left') ? (e.clientX - startX) : (startX - e.clientX);
                    var newW = Math.min(maxW, Math.max(minW, startW + delta));
                    sidebar.style.width = newW + 'px';
                    sidebar.style.minWidth = newW + 'px';
                }
                function onUp() {
                    document.removeEventListener('mousemove', onMove);
                    document.removeEventListener('mouseup', onUp);
                }
                document.addEventListener('mousemove', onMove);
                document.addEventListener('mouseup', onUp);
            });
        }
        initResize('left-resize-handle', 'left-sidebar', 'left');
        initResize('right-resize-handle', 'right-sidebar', 'right');
        return window.dash_clientside.no_update;
    }
    """,
    Output("sidebar-resize-dummy", "data"),
    Input("left-resize-handle", "id"),
    Input("right-resize-handle", "id"),
    prevent_initial_call=False,
)


# ── 모드 2 아코디언: 체크박스 → 바디 표시/숨김 ───────────────────────────

_MODE2_ALGO_NAMES = json.dumps([cls.name for cls in REGISTRY])
_MODE2_BODY_OPEN_STYLE = json.dumps({
    "display": "block", "padding": "8px 10px 6px",
    "border": "1px solid rgba(133, 91, 251, 0.40)", "borderTop": "none",
    "borderBottomLeftRadius": "6px", "borderBottomRightRadius": "6px",
    "background": "#ffffff",
})

app.clientside_callback(
    f"""
    function(check_values, check_ids) {{
        var names = {_MODE2_ALGO_NAMES};
        var open_style = {_MODE2_BODY_OPEN_STYLE};
        var checked = {{}};
        for (var i = 0; i < check_ids.length; i++) {{
            checked[check_ids[i].algo] = (check_values[i] || []).indexOf("on") !== -1;
        }}
        return names.map(function(name) {{
            return checked[name] ? open_style : {{display: "none"}};
        }});
    }}
    """,
    [Output({"type": "mode2-body", "algo": cls.name}, "style") for cls in REGISTRY],
    Input({"type": "mode2-algo-check", "algo": ALL}, "value"),
    State({"type": "mode2-algo-check", "algo": ALL}, "id"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    test_port = os.environ.get("DASH_PORT")
    if test_port:
        app.run(debug=False, port=int(test_port), host="127.0.0.1", use_reloader=False)
    else:
        app.run(debug=True)
