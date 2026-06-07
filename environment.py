import math
import numpy as np
import pandas as pd
from typing import Iterable
from scipy.stats import multivariate_normal
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.vectorized import contains


def _to_polygon_list(geometry: object) -> list[Polygon]:
    if geometry is None:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [g for g in geometry.geoms if isinstance(g, Polygon)]
    if isinstance(geometry, GeometryCollection):
        out: list[Polygon] = []
        for g in geometry.geoms:
            out.extend(_to_polygon_list(g))
        return out
    if isinstance(geometry, LineString):
        buffered = geometry.buffer(0.00005)
        if isinstance(buffered, Polygon):
            return [buffered]
    if isinstance(geometry, MultiLineString):
        out = []
        for g in geometry.geoms:
            out.extend(_to_polygon_list(g))
        return out
    if isinstance(geometry, (list, tuple, set)):
        out = []
        for g in geometry:
            out.extend(_to_polygon_list(g))
        return out
    return []

class SyntheticEnvironment:
    """
    1. Local Coordinate: 좌상단을 (0,0)으로 하는 미터(m) 단위 좌표계
        -> 좌상단 (0, 0), 좌하단 (0, height_m), 우상단 (width_m, 0), 우하단 (width_m, height_m)
    2. Geo Coordinate: 실제 위도/경도 좌표계
    """

    def __init__(self, center_lat=37.4979, center_lon=127.0276, width_km=2.0, height_km=None, resolution_m=100):
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        if height_km is None: # 이제 이거 없어도 됨. 근데 오류날 거 같으니까 안뺌.
            height_km = width_km
            
        self.width_km = width_km
        self.height_km = height_km
        
        self.width_m = width_km * 1000
        self.height_m = height_km * 1000
        self.resolution_m = resolution_m
        
        # 격자 개수 계산 (가로, 세로 다를 수 있음)
        self.cols = int(self.width_m / resolution_m)  # x축 격자 수
        self.rows = int(self.height_m / resolution_m) # y축 격자 수
        
        self.x_range = np.linspace(0, self.width_m, self.cols)
        self.y_range = np.linspace(0, self.height_m, self.rows)
        
        # meshgrid: x는 cols개, y는 rows개
        self.x_grid, self.y_grid = np.meshgrid(self.x_range, self.y_range)
        
        lat_span = self.height_km / 111.0
        lon_span = self.width_km / 88.0
        
        self.lat_min = center_lat - lat_span / 2
        self.lat_max = center_lat + lat_span / 2
        self.lon_min = center_lon - lon_span / 2
        self.lon_max = center_lon + lon_span / 2
        
        self.lat_range = np.linspace(self.lat_min, self.lat_max, self.rows)
        self.lon_range = np.linspace(self.lon_min, self.lon_max, self.cols)
        
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon_range, self.lat_range)
        
        # 데이터 저장소
        self.traffic_map = np.zeros((self.rows, self.cols))
        self.obstacles = []
        self.obstacles_geo = []
        self.station_candidate_points: list[tuple[float, float]] = []

        self._raw_traffic_map = self.traffic_map.copy()
        self._raw_traffic_series = None
        self.traffic_series = None
        self.dynamic_frame_index = 0
        self._obstacle_mask = np.zeros((self.rows, self.cols), dtype=bool)

    def generate_traffic(self, num_hotspots=5, max_intensity=100, spread_m=500, base_intensity=10):
        """레거시 multi_hotspot 생성 (m 단위 spread). 기존 동작 유지."""
        # 1. Base Traffic 생성: 랜덤값 추가.
        noise = np.random.uniform(0, 20, (self.rows, self.cols))
        self.traffic_map = np.full((self.rows, self.cols), base_intensity) + noise

        # 2. 핫스팟 추가
        for _ in range(num_hotspots):
            h_x = np.random.choice(self.x_range)
            h_y = np.random.choice(self.y_range)

            pos = np.dstack((self.x_grid, self.y_grid))

            safe_spread = max(spread_m, 10)
            rv = multivariate_normal([h_x, h_y], [[safe_spread**2, 0], [0, safe_spread**2]])

            pdf_values = rv.pdf(pos)

            if pdf_values.max() > 0:
                self.traffic_map += pdf_values * (max_intensity / pdf_values.max())

        self._raw_traffic_map = self.traffic_map.copy()
        self.traffic_series = None
        self._raw_traffic_series = None
        self.dynamic_frame_index = 0
        self.remask_traffic()
        return self.traffic_map

    def generate_traffic_pattern(self, pattern: str, max_intensity: float = 100.0,
                                  base_intensity: float = 10.0, params: dict | None = None,
                                  rng: 'np.random.Generator | None' = None):
        """bs_opt에서 포팅된 8종 패턴 생성기. patterns.generate_pattern 사용.

        patterns.py가 [0, 1] 정규화된 맵을 반환하면, 여기서 base_intensity + max_intensity·t로
        기존 스케일과 호환되게 변환한다.

        Args:
            pattern: patterns.PATTERN_CHOICES 중 하나
            max_intensity: 정규화된 최댓값이 가질 스케일
            base_intensity: 오프셋 (모든 셀에 더해짐)
            params: 패턴별 세부 하이퍼파라미터 (patterns.py 참조)
            rng: numpy Generator (None이면 전역 random 사용)
        """
        from patterns import generate_pattern
        if rng is None:
            rng = np.random.default_rng()
        normalized = generate_pattern(self.rows, self.cols, pattern=pattern,
                                      rng=rng, params=params)
        self.traffic_map = base_intensity + normalized * max_intensity
        self._raw_traffic_map = self.traffic_map.copy()
        self.traffic_series = None
        self._raw_traffic_series = None
        self.dynamic_frame_index = 0
        self.remask_traffic()
        return self.traffic_map

    def generate_dynamic_traffic_pattern(self, pattern: str, time_steps=12, max_intensity: float = 100.0,
                                        base_intensity: float = 10.0, variation: float = 0.25,
                                        drift_m: float = 300.0, params: dict | None = None):
        """시간축 트래픽 생성기.

        패턴/강도/노이즈를 프레임마다 변조해 시계열 트래픽 맵을 만든다.
        """
        from patterns import generate_pattern

        params = dict(params or {})
        time_steps = max(1, int(time_steps))
        variation = float(variation)
        drift_cells = max(0.0, drift_m / max(float(self.resolution_m), 1.0))
        rng = np.random.default_rng()

        centers = None
        directions = None
        if pattern == "multi_hotspot":
            n_centers = int(params.get("n_centers", 5))
            centers = params.get("centers")
            if centers is None:
                centers = [
                    (rng.uniform(0, self.cols - 1), rng.uniform(0, self.rows - 1))
                    for _ in range(max(1, n_centers))
                ]
            centers = [tuple(map(float, c)) for c in centers]
            if len(centers) == 0:
                centers = [
                    (self.cols / 2.0, self.rows / 2.0),
                ]
            raw_dirs = rng.normal(size=(len(centers), 2))
            norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            directions = raw_dirs / norms

        frames = []
        for step in range(time_steps):
            phase = step / max(1, time_steps - 1)
            frame_seed = int(rng.integers(0, 2**31 - 1))
            frame_rng = np.random.default_rng(frame_seed)
            frame_params = dict(params)

            if pattern == "multi_hotspot" and centers is not None and directions is not None:
                shifted = []
                amp = math.sin(2.0 * np.pi * phase)
                for (cx, cy), (dx, dy) in zip(centers, directions):
                    sx = cx + dx * drift_cells * amp
                    sy = cy + dy * drift_cells * amp
                    sx = float(np.clip(sx, 0, self.cols - 1))
                    sy = float(np.clip(sy, 0, self.rows - 1))
                    shifted.append((sx, sy))
                frame_params["centers"] = shifted
                frame_params.setdefault("n_centers", len(shifted))

            if pattern == "ring":
                base_radius = float(params.get("radius", min(self.cols, self.rows) / 3.0))
                frame_params["radius"] = max(1.0, base_radius + (drift_cells * 0.3) * np.sin(2.0 * np.pi * phase))

            if pattern == "stripe":
                stripe_pos = params.get("stripe_pos", int(min(self.cols, self.rows) / 2))
                stripe_pos = int(stripe_pos + drift_cells * np.sin(2.0 * np.pi * phase))
                if frame_params.get("orientation", "vertical") == "vertical":
                    frame_params["stripe_pos"] = np.clip(stripe_pos, 0, self.cols - 1)
                else:
                    frame_params["stripe_pos"] = np.clip(stripe_pos, 0, self.rows - 1)

            if pattern == "gradient":
                frame_params["direction"] = "ew" if phase < 0.5 else "ns"

            if pattern == "random_clusters":
                frame_params["n_clusters"] = int(params.get("n_clusters", 5))

            frame_norm = generate_pattern(
                self.rows,
                self.cols,
                pattern=pattern,
                rng=frame_rng,
                params=frame_params,
            )
            frame_norm = np.clip(frame_norm * (1.0 + variation * np.sin(2.0 * np.pi * phase)), 0.0, 1.0)
            frame_shift = int(round(drift_cells * np.cos(2.0 * np.pi * phase)))
            if frame_shift != 0:
                frame_norm = np.roll(frame_norm, frame_shift, axis=1)
            frame = base_intensity + frame_norm * max_intensity
            frames.append(frame.astype(float))

        self._raw_traffic_series = np.stack(frames, axis=0)
        self.traffic_series = self._raw_traffic_series.copy()
        self._raw_traffic_map = self._raw_traffic_series[0].copy()
        self.traffic_map = self._raw_traffic_map.copy()
        self.dynamic_frame_index = 0
        self.remask_traffic()
        return self.traffic_series

    def generate_obstacles(self, num_obstacles=5, pattern='random'):
        # 장애물 생성 (겹침 방지 포함)
        # pattern: 'random', 'circle', 'strip', 'grid', 'mixed'
        self.obstacles = []
        max_attempts = 50 # 겹치지 않는 위치를 찾기 위한 최대 시도 횟수
        
        count = 0
        while count < num_obstacles:
            current_pattern = pattern
            if pattern == 'mixed':
                current_pattern = np.random.choice(['random', 'circle', 'strip', 'grid'])
            
            # 장애물 후보 생성
            candidate = None
            for _ in range(max_attempts):
                temp_obs = self._create_single_obstacle(current_pattern)
                
                # 겹치는지 확인하고
                intersects = False
                for existing in self.obstacles:
                    if temp_obs.intersects(existing):
                        intersects = True
                        break
                
                if not intersects:
                    candidate = temp_obs
                    break
            
            # 안겹치면 추가
            if candidate:
                self.obstacles.append(candidate)
                count += 1
            else:
                # 너무 빽빽해서 더 이상 추가 불가능하면 중단할 수도 있음
                # 여기서는 그냥 넘어가고 계속 시도 (혹은 break)
                print(f"Warning: Could not place obstacle {count+1} after {max_attempts} attempts.")
                break
        
        self._convert_obstacles_to_geo()
        return self.obstacles

    def _create_single_obstacle(self, pattern):
        if pattern == 'random': # 다각형
            cx, cy = np.random.choice(self.x_range), np.random.choice(self.y_range)
            radius = np.random.uniform(100, min(self.width_m, self.height_m) * 0.1)
            points = []
            for angle in np.linspace(0, 2*np.pi, 10, endpoint=False):
                r = radius * np.random.uniform(0.5, 1.5)
                points.append((cx + r*np.cos(angle), cy + r*np.sin(angle)))
            return Polygon(points)
            
        elif pattern == 'circle':
            cx, cy = np.random.choice(self.x_range), np.random.choice(self.y_range)
            radius = np.random.uniform(150, min(self.width_m, self.height_m) * 0.15)
            return Point(cx, cy).buffer(radius)
            
        elif pattern == 'strip':
            w = np.random.uniform(50, 150)
            if np.random.random() > 0.5: # 수평
                cy = np.random.choice(self.y_range)
                return box(0, cy - w/2, self.width_m, cy + w/2)
            else: # 수직
                cx = np.random.choice(self.x_range)
                return box(cx - w/2, 0, cx + w/2, self.height_m)
                
        elif pattern == 'grid':
            # 격자는 위치를 랜덤하게 잡아서 하나만 생성
            block_size = 200
            x = np.random.choice(self.x_range)
            y = np.random.choice(self.y_range)
            return box(x, y, x + block_size, y + block_size)
            
        return None

    def _convert_obstacles_to_geo(self):
        # Local -> Geo 좌표 변환
        self.obstacles_geo = []
        
        x_scale = (self.lon_max - self.lon_min) / self.width_m
        y_scale = (self.lat_max - self.lat_min) / self.height_m
        
        for poly in self.obstacles:
            coords = list(poly.exterior.coords)
            new_coords = []
            for x, y in coords:
                new_lon = self.lon_min + x * x_scale
                new_lat = self.lat_min + y * y_scale
                new_coords.append((new_lon, new_lat))
            
            self.obstacles_geo.append(Polygon(new_coords))

    def geo_to_local_polygons(self, geometry: object) -> list[Polygon]:
        x_scale = self.width_m / (self.lon_max - self.lon_min)
        y_scale = self.height_m / (self.lat_max - self.lat_min)
        local_polygons = []

        def _to_local_point(xy):
            x, y = xy
            return (x - self.lon_min) * x_scale, (y - self.lat_min) * y_scale

        for poly in _to_polygon_list(geometry):
            try:
                exterior = [_to_local_point(c) for c in poly.exterior.coords]
                interiors = []
                for interior in poly.interiors:
                    interiors.append([_to_local_point(c) for c in interior.coords])
                candidate = Polygon(exterior, interiors)
                if candidate.is_empty:
                    continue
                if not candidate.is_valid:
                    candidate = candidate.buffer(0)
                if candidate.is_empty:
                    continue
                local_polygons.append(candidate)
            except Exception:
                continue
        return local_polygons

    def local_points_to_geo(self, points):
        if points is None:
            return np.empty((0, 2))
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2))
        if arr.ndim == 1 and arr.shape == (2,):
            arr = arr.reshape(1, 2)
        if arr.ndim != 2:
            return np.empty((0, 2))

        x_scale = (self.lon_max - self.lon_min) / self.width_m
        y_scale = (self.lat_max - self.lat_min) / self.height_m

        lons = self.lon_min + arr[:, 0] * x_scale
        lats = self.lat_min + arr[:, 1] * y_scale
        return np.column_stack((lats, lons))

    def set_station_candidate_points(self, points):
        if points is None:
            self.station_candidate_points = []
            return
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            self.station_candidate_points = []
            return
        if arr.ndim == 1 and arr.shape == (2,):
            arr = arr.reshape(1, 2)
        self.station_candidate_points = [tuple(p.tolist()) for p in arr]

    def append_station_candidate_points(self, points):
        if points is None:
            return
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return
        if arr.ndim == 1 and arr.shape == (2,):
            arr = arr.reshape(1, 2)
        self.station_candidate_points.extend([tuple(p.tolist()) for p in arr])

    def clear_station_candidate_points(self):
        self.station_candidate_points = []

    def append_obstacles(self, polygons: Iterable[Polygon]):
        for polygon in _to_polygon_list(polygons):
            if polygon is None or polygon.is_empty:
                continue
            if polygon.area <= 0:
                continue
            self.obstacles.append(polygon)
        self._convert_obstacles_to_geo()
        self._obstacle_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.remask_traffic()

    def replace_obstacles(self, polygons: Iterable[Polygon]):
        self.obstacles = []
        self.append_obstacles(polygons)
        self.remask_traffic()

    def clear_obstacles(self):
        self.obstacles = []
        self.obstacles_geo = []
        self._obstacle_mask = np.zeros((self.rows, self.cols), dtype=bool)
        self.remask_traffic()

    def remask_traffic(self):
        current = self._get_raw_active_map()
        if current is None:
            return
        self.traffic_map = np.array(current, copy=True)

    def apply_masking(self):
        # 장애물 지도에서 마스킹하기
        if not self.obstacles:
            return self.traffic_map
            
        flat_x = self.x_grid.ravel()
        flat_y = self.y_grid.ravel()
        mask = np.zeros(flat_x.shape, dtype=bool)
        
        for poly in self.obstacles:
            in_poly = contains(poly, flat_x, flat_y)
            mask = mask | in_poly
            
        self.traffic_map.ravel()[mask] = 0
        return self.traffic_map

    def get_dataframe(self):
        # Geo 좌표계 DF 반환
        df = pd.DataFrame({
            'lat': self.lat_grid.ravel(),
            'lon': self.lon_grid.ravel(),
            'traffic': self.traffic_map.ravel()
        })
        return df[df['traffic'] > 0.1]

    def get_local_data(self):
        # Local 좌표계 Array 반환 (좌하단 기준)
        if self._raw_traffic_map is None:
            traffic = self.traffic_map
        else:
            traffic = self._get_raw_active_map()
            if traffic is None:
                traffic = self.traffic_map
        mask = np.asarray(traffic).ravel() > 0.1
        if self.obstacles:
            mask &= ~self.get_obstacle_mask().ravel()
        
        x_vals = self.x_grid.ravel()[mask]
        y_vals = self.y_grid.ravel()[mask]
        traffic_vals = np.asarray(traffic).ravel()[mask]
        
        return np.column_stack((x_vals, y_vals, traffic_vals))
    
    def get_local_data_top_left(self):
        # Local 좌표계 Array 반환 (좌상단 기준)
        # x: 그대로 (Left -> Right)
        # y: 반전 (Top -> Bottom, 즉 y=0이 최상단)
        if self._raw_traffic_map is None:
            traffic = self.traffic_map
        else:
            traffic = self._get_raw_active_map()
            if traffic is None:
                traffic = self.traffic_map
        mask = np.asarray(traffic).ravel() > 0.1
        if self.obstacles:
            mask &= ~self.get_obstacle_mask().ravel()
        
        x_vals = self.x_grid.ravel()[mask]
        y_vals_original = self.y_grid.ravel()[mask]
        traffic_vals = np.asarray(traffic).ravel()[mask]
        
        y_vals_inverted = self.height_m - y_vals_original
        
        return np.column_stack((x_vals, y_vals_inverted, traffic_vals))

    def _get_raw_active_map(self):
        if self._raw_traffic_series is not None:
            if self.traffic_series is None or self.traffic_series.ndim != 3:
                return None
            idx = int(self.dynamic_frame_index)
            if idx < 0 or idx >= self.traffic_series.shape[0]:
                idx = 0
            return self._raw_traffic_series[idx]
        if self._raw_traffic_map is not None:
            return self._raw_traffic_map
        return None

    def get_raw_traffic_map(self):
        current = self._get_raw_active_map()
        if current is not None:
            return current
        return self.traffic_map

    def get_raw_traffic_series(self):
        return self._raw_traffic_series

    def set_traffic_frame(self, frame_idx: int):
        if self._raw_traffic_series is None:
            self.dynamic_frame_index = 0
            return
        if self.traffic_series is None or self.traffic_series.ndim != 3 or self.traffic_series.shape[0] <= 1:
            self.dynamic_frame_index = 0
            return
        idx = int(frame_idx)
        idx = max(0, min(idx, self.traffic_series.shape[0] - 1))
        self.dynamic_frame_index = idx
        self.remask_traffic()

    def get_obstacle_mask(self):
        self._obstacle_mask = np.zeros((self.rows, self.cols), dtype=bool)
        if not self.obstacles:
            self._obstacle_mask[:] = False
            return self._obstacle_mask

        flat_x = self.x_grid.ravel()
        flat_y = self.y_grid.ravel()
        mask = np.zeros(flat_x.shape, dtype=bool)
        for poly in self.obstacles:
            if poly is None or poly.is_empty:
                continue
            in_poly = contains(poly, flat_x, flat_y)
            mask = mask | in_poly

        self._obstacle_mask = mask.reshape(self.rows, self.cols)
        return self._obstacle_mask
