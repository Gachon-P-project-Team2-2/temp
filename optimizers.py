import numpy as np
from sklearn.cluster import KMeans
import copy

class BaseStationOptimizer:
    def __init__(self, env, radius=300, capacity=1000):
        """
        Args:
            env: SyntheticEnvironment 객체
            radius (float): 기지국 커버리지 반경 (m)
            capacity (float): 기지국 하나당 처리 가능한 최대 트래픽 양
        """
        self.env = env
        self.radius = radius
        self.capacity = capacity
        
        # Local 데이터 로드 (x, y, traffic)
        self.data = env.get_local_data()
        
        # X: (N, 2) 형태의 좌표 배열 [x, y]
        self.X = self.data[:, 0:2]
        # weights: (N,) 형태의 트래픽 배열
        self.weights = self.data[:, 2]
        
        # 탐색 범위 (경계) 설정
        self.x_min, self.x_max = 0, env.width_m
        self.y_min, self.y_max = 0, env.height_m

    def _calculate_score(self, centers):
        if len(centers) == 0: return 0
        
        diff = self.X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        radius_sq = self.radius ** 2
        
        covered_mask = dist_sq <= radius_sq
        is_covered = np.any(covered_mask, axis=1)
        
        dist_sq_masked = np.where(covered_mask, dist_sq, np.inf)
        nearest_station_idx = np.argmin(dist_sq_masked, axis=1)
        
        station_loads = np.zeros(len(centers))
        
        valid_indices = np.where(is_covered)[0]
        if len(valid_indices) > 0:
            assigned_stations = nearest_station_idx[valid_indices]
            traffic_values = self.weights[valid_indices]
            np.add.at(station_loads, assigned_stations, traffic_values)
            
            effective_loads = np.minimum(station_loads, self.capacity)
            
            total_covered_traffic = np.sum(effective_loads)
            total_covered_area = len(valid_indices)
            
            return total_covered_traffic + (total_covered_area * 0.1)
            
        return 0

    def get_stats(self, centers):
        if len(centers) == 0: return {}
        
        # 1. 거리 계산
        diff = self.X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        radius_sq = self.radius ** 2
        
        # 2. 커버리지 확인
        covered_mask = dist_sq <= radius_sq
        is_covered = np.any(covered_mask, axis=1)
        
        # 3. 할당 (가장 가까운 유효 기지국)
        dist_sq_masked = np.where(covered_mask, dist_sq, np.inf)
        nearest_station_idx = np.argmin(dist_sq_masked, axis=1)
        
        # 4. 부하 계산
        station_loads = np.zeros(len(centers))
        
        valid_indices = np.where(is_covered)[0]
        if len(valid_indices) > 0:
            assigned_stations = nearest_station_idx[valid_indices]
            traffic_values = self.weights[valid_indices]
            np.add.at(station_loads, assigned_stations, traffic_values)
        
        # 실제 커버된 트래픽 (용량 제한 적용)
        effective_loads = np.minimum(station_loads, self.capacity)
        
        stats = {
            'total_traffic': np.sum(self.weights),
            'covered_traffic': np.sum(effective_loads),
            'total_area': len(self.weights), # 전체 격자 수
            'covered_area': len(valid_indices), # 커버된 격자 수
            'station_loads': station_loads, # 기지국별 요청 트래픽
            'station_effective_loads': effective_loads, # 기지국별 실제 처리 트래픽
            'capacity': self.capacity
        }
        return stats

    def convert_to_geo(self, centers_local):
        # Local -> Geo 좌표 변환
        x_scale = (self.env.lon_max - self.env.lon_min) / self.env.width_m
        y_scale = (self.env.lat_max - self.env.lat_min) / self.env.height_m
        
        lon_origin = self.env.lon_min
        lat_origin = self.env.lat_min
        
        centers_geo = []
        for x, y in centers_local:
            lon = lon_origin + x * x_scale
            lat = lat_origin + y * y_scale
            centers_geo.append([lat, lon])
            
        return np.array(centers_geo)

    def run_kmeans(self, n_stations, n_init=10, random_state=42):
        # random_state=-1 이면 매 실행마다 다른 결과 (시드 미고정)
        rs = None if random_state == -1 else random_state
        kmeans = KMeans(n_clusters=n_stations, n_init=n_init, random_state=rs)
        kmeans.fit(self.X, sample_weight=self.weights)
        centers = kmeans.cluster_centers_
        score = self._calculate_score(centers)
        return centers, score

    def run_random_walk(self, n_stations, iterations=1000, step_size=50.0):
        # 초기화: _get_random_centers(n_stations)
        # 반복: iterations 횟수만큼 랜덤 워크로 기지국 위치 조정 -> 점수 계산 -> 최적 위치 찾기
        # 점수: _calculate_score(current_centers)
        # 반환: best_centers, best_score
        
        current_centers = self._get_random_centers(n_stations)
        current_score = self._calculate_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        
        for _ in range(iterations):
            noise = np.random.normal(0, step_size, size=current_centers.shape)
            next_centers = current_centers + noise
            self._clip_centers(next_centers)
            next_score = self._calculate_score(next_centers)
            if next_score > current_score:
                current_centers = next_centers
                current_score = next_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        return best_centers, best_score

    def run_simulated_annealing(self, n_stations, iterations=1000, initial_temp=100.0, cooling_rate=0.99, step_size=50.0):
        current_centers = self._get_random_centers(n_stations)
        current_score = self._calculate_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        temp = initial_temp
        
        for _ in range(iterations):
            noise = np.random.normal(0, step_size, size=current_centers.shape)
            next_centers = current_centers + noise
            self._clip_centers(next_centers)
            next_score = self._calculate_score(next_centers)
            delta = current_score - next_score 
            if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-5)):
                current_centers = next_centers
                current_score = next_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
            temp *= cooling_rate
        return best_centers, best_score

    def run_tabu_search(self, n_stations, iterations=500, step_size=50.0, tabu_tenure=10):
        current_centers = self._get_random_centers(n_stations)
        current_score = self._calculate_score(current_centers)
        best_centers = current_centers.copy()
        best_score = current_score
        tabu_list = {} 
        for it in range(iterations):
            candidates = []
            for _ in range(20):
                idx = np.random.randint(0, n_stations)
                if idx in tabu_list and tabu_list[idx] > it: continue
                cand = current_centers.copy()
                cand[idx] += np.random.normal(0, step_size, 2)
                self._clip_centers(cand)
                score = self._calculate_score(cand)
                candidates.append((score, cand, idx))
            if not candidates: continue
            candidates.sort(key=lambda x: x[0], reverse=True)
            chosen_score, chosen_centers, moved_idx = candidates[0]
            current_centers = chosen_centers
            current_score = chosen_score
            tabu_list[moved_idx] = it + tabu_tenure
            if current_score > best_score:
                best_score = current_score
                best_centers = current_centers.copy()
        return best_centers, best_score

    def _get_random_centers(self, k):
        # 랜덤 초기 기지국 위치 생성
        x = np.random.uniform(self.x_min, self.x_max, k)
        y = np.random.uniform(self.y_min, self.y_max, k)
        return np.column_stack([x, y])
    
    def _clip_centers(self, centers):
        # 기지국 위치 클리핑 (경계 초과안하게)
        centers[:, 0] = np.clip(centers[:, 0], self.x_min, self.x_max)
        centers[:, 1] = np.clip(centers[:, 1], self.y_min, self.y_max)
