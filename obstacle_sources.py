"""오브젝트 소스 로더 유틸리티.

앱(`app.py`)에서 필요한 공개 함수:
- geojson_to_polygons
- filter_polygons
- load_osm_polygons_with_cache
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Iterable

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union


_CACHE_TTL_SECONDS = 60 * 60 * 24
_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "osm_polygons")
_OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"


def _ensure_cache_dir() -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(lat_min: float, lon_min: float, lat_max: float, lon_max: float, types: list[str]) -> str:
    payload = json.dumps(
        {
            "bbox": [lat_min, lon_min, lat_max, lon_max],
            "types": sorted(set(types)),
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cache(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: str, data: dict[str, Any]) -> None:
    _ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _extract_polygons_from_geometry(geo_obj: Any) -> list[Polygon]:
    if geo_obj is None:
        return []
    if isinstance(geo_obj, Polygon):
        return [geo_obj]
    if isinstance(geo_obj, MultiPolygon):
        return [g for g in geo_obj.geoms if isinstance(g, Polygon)]
    if isinstance(geo_obj, GeometryCollection):
        out: list[Polygon] = []
        for item in geo_obj.geoms:
            out.extend(_extract_polygons_from_geometry(item))
        return out
    try:
        parsed = shape(geo_obj)
    except Exception:
        return []
    if isinstance(parsed, Polygon):
        return [parsed]
    if isinstance(parsed, MultiPolygon):
        return [g for g in parsed.geoms if isinstance(g, Polygon)]
    if isinstance(parsed, GeometryCollection):
        out = []
        for item in parsed.geoms:
            out.extend(_extract_polygons_from_geometry(item))
        return out
    # 선형 오브젝트는 좁은 버퍼로 근사 폴리곤화
    if isinstance(parsed, LineString):
        buffered = parsed.buffer(0.00005)
        return [buffered] if isinstance(buffered, Polygon) and buffered.is_valid else []
    if hasattr(parsed, "geoms"):
        out = []
        for item in getattr(parsed, "geoms", []):
            out.extend(_extract_polygons_from_geometry(item))
        return out
    return []


def _normalize_polygon_area_to_m2(poly: Polygon, lat_ref: float | None = None) -> float:
    if not isinstance(poly, Polygon) or poly.is_empty:
        return 0.0
    lat_ref = float(lat_ref if lat_ref is not None else poly.centroid.y)
    deg_to_m_lat = 110_540.0
    deg_to_m_lon = 111_320.0 * math.cos(math.radians(lat_ref))
    return float(poly.area * deg_to_m_lat * deg_to_m_lon)


def _parse_geojson_payload(payload: str | bytes) -> list[Polygon]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    geoms: list[Any] = []
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            geom = feature.get("geometry")
            if geom:
                geoms.append(geom)
    elif isinstance(data, dict) and "geometry" in data:
        geoms.append(data.get("geometry"))
    elif isinstance(data, dict) and "elements" in data:
        # Overpass 응답은 Element 형식일 수 있음
        for element in data.get("elements", []):
            geometry = element.get("geometry")
            if geometry:
                geoms.append({"type": "LineString", "coordinates": [[pt["lon"], pt["lat"]] for pt in geometry]})
    elif isinstance(data, list):
        geoms.extend(data)
    out: list[Polygon] = []
    for g in geoms:
        out.extend(_extract_polygons_from_geometry(g))
    return out


def geojson_to_polygons(raw: str | bytes | dict | list) -> list[Polygon]:
    """GeoJSON 텍스트(또는 dict/list)에서 다각형을 추출한다."""
    if isinstance(raw, (dict, list)):
        raw = json.dumps(raw, ensure_ascii=False)
    if not raw:
        return []
    try:
        polygons = _parse_geojson_payload(raw)
    except Exception:
        return []
    out: list[Polygon] = []
    for poly in polygons:
        if poly.is_empty:
            continue
        fixed = poly.buffer(0)
        if fixed.is_empty:
            continue
        if isinstance(fixed, MultiPolygon):
            out.extend([p for p in fixed.geoms if isinstance(p, Polygon) and not p.is_empty])
        elif isinstance(fixed, Polygon):
            out.append(fixed)
        elif isinstance(fixed, GeometryCollection):
            for p in _extract_polygons_from_geometry(fixed):
                if isinstance(p, Polygon) and not p.is_empty:
                    out.append(p)
    return out


def filter_polygons(
    polygons: Iterable[Polygon],
    min_area_m2: float,
    max_obstacles: int | None = None,
) -> list[Polygon]:
    """면적 임계치/개수 조건으로 폴리곤을 필터한다."""
    filtered: list[tuple[Polygon, float]] = []
    for poly in polygons:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        fixed = poly.buffer(0)
        if fixed.is_empty:
            continue
        area_m2 = _normalize_polygon_area_to_m2(fixed)
        if min_area_m2 is not None and min_area_m2 > 0 and area_m2 < min_area_m2:
            continue
        filtered.append((fixed, area_m2))
    filtered.sort(key=lambda item: item[1], reverse=True)
    polygons_sorted = [poly for poly, _ in filtered]
    if max_obstacles is None or max_obstacles <= 0:
        return polygons_sorted
    return polygons_sorted[:max_obstacles]


def _canonicalize_obstacle_types(types: list[str] | None) -> list[str]:
    if not types:
        return ["building", "water", "waterway", "highway"]
    out: list[str] = []
    for t in types:
        if t == "road":
            out.append("highway")
        else:
            out.append(t)
    dedup = []
    for t in out:
        if t not in dedup:
            dedup.append(t)
    return dedup


def _build_overpass_query(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    obstacle_types: list[str],
) -> str:
    west, south, east, north = lon_min, lat_min, lon_max, lat_max
    clauses = []
    for t in obstacle_types:
        if t == "building":
            clauses.append(f'way["building"]({south},{west},{north},{east});')
        elif t == "water":
            clauses.append(f'way["water"]({south},{west},{north},{east});')
            clauses.append(f'way["natural"="water"]({south},{west},{north},{east});')
        elif t == "waterway":
            clauses.append(f'way["waterway"]({south},{west},{north},{east});')
        elif t == "highway":
            clauses.append(f'way["highway"]({south},{west},{north},{east});')
        else:
            clauses.append(f'way["{t}"]({south},{west},{north},{east});')
    if not clauses:
        clauses = ['way["building"](...);']
    joined = "".join(clauses)
    return f"[out:json][timeout:25];({joined});(._;>;);out geom;"


def load_osm_polygons_with_cache(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    obstacle_types: list[str] | None = None,
) -> tuple[list[Polygon], int]:
    """지역 bbox 내 OSM 다각형을 조회해 폴리곤 리스트와 원본 개수를 반환한다."""
    obstacle_types = _canonicalize_obstacle_types(list(obstacle_types or []))
    cache_key = _cache_key(lat_min, lon_min, lat_max, lon_max, obstacle_types)
    cache_file = os.path.join(_CACHE_DIR, f"{cache_key}.json")

    data = _load_cache(cache_file)
    if data is None:
        query = _build_overpass_query(lat_min, lon_min, lat_max, lon_max, obstacle_types)
        payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
        request = urllib.request.Request(_OVERPASS_ENDPOINT, data=payload, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                text = response.read().decode("utf-8")
            parsed = json.loads(text)
            data = parsed
            _save_cache(cache_file, parsed)
        except Exception:
            # 네트워크 장애 또는 쿼리 실패 시 폴백: 빈 결과 반환
            return [], 0
    else:
        # 캐시 적재
        parsed = data

    polygons = _parse_osm_payload(parsed)
    raw_count = len(polygons)
    return polygons, raw_count


def _parse_osm_payload(data: dict[str, Any]) -> list[Polygon]:
    if not isinstance(data, dict):
        return []
    polygons: list[Polygon] = []
    elements = data.get("elements", [])
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        geometry = elem.get("geometry")
        if not geometry or not isinstance(geometry, list):
            continue
        coords = [[float(pt.get("lon", 0.0)), float(pt.get("lat", 0.0))] for pt in geometry if isinstance(pt, dict) and "lon" in pt and "lat" in pt]
        if len(coords) < 2:
            continue
        geom = None
        if len(coords) >= 4 and coords[0] == coords[-1]:
            geom = Polygon(coords)
        else:
            geom = LineString(coords).buffer(0.00005)
        if geom is None or geom.is_empty:
            continue
        polygons.extend(_extract_polygons_from_geometry(geom))
    merged = unary_union(polygons) if polygons else None
    if merged is None:
        return []
    return _extract_polygons_from_geometry(merged)
