import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput
from optimizers.metaheuristics._shared import calculate_score, snap_stations_to_candidates
from optimizers.kmeans import KMeansOptimizer
from optimizers.drl.dqn_placement import DQNPlacementOptimizer
from sklearn.cluster import KMeans

import logging
logging.getLogger("optimizers").setLevel(logging.CRITICAL)

def main():
    # 1. 환경 설정 (Seed 고정)
    env = SyntheticEnvironment(width_km=2.0, height_km=2.0, resolution_m=50)
    rng = np.random.default_rng(42)
    env.generate_traffic_pattern_density(area_demand_mbps_km2=100.0, pattern="multi_hotspot", rng=rng)
    
    n_stations = 5
    problem = ProblemInput.from_env(
        env,
        radius_m=500.0,
        capacity=1e10,
        tx_power_dbm=43.0,
        bandwidth_mhz=10.0,
        score_mode="traffic"
    )
    
    # 1. K-Means 실행 결과
    km_opt = KMeansOptimizer()
    km_res = km_opt.optimize(problem, n_stations=n_stations, random_state=42)
    km_score = km_res.score
    
    # DQN 내부의 warm_start_stations 계산 로직 동일 재현
    km = KMeans(n_clusters=n_stations, n_init=10, random_state=42)
    km.fit(problem.X, sample_weight=problem.weights)
    warm_start_stations = snap_stations_to_candidates(km.cluster_centers_, problem)
    warm_score = calculate_score(warm_start_stations, problem)
    
    # 2. DQN 실행 결과 (best_stations)
    dqn_opt = DQNPlacementOptimizer()
    dqn_res = dqn_opt.optimize(problem, n_stations=n_stations, episodes=50, steps_per_episode=20, random_state=42)
    best_stations = dqn_res.stations
    best_score = calculate_score(best_stations, problem)
    
    # 출력
    print("1. K-Means 실행 결과 score:")
    print(f"{km_score:.4f}")
    
    print("\n2. DQN warm_start_stations score:")
    print(f"{warm_score:.4f}")
    
    print("\n3. DQN best_stations score:")
    print(f"{best_score:.4f}")
    
    print("\nnp.allclose(best_stations, warm_start_stations):")
    print(np.allclose(best_stations, warm_start_stations))
    
    print("\nbest_stations 좌표:")
    print(repr(best_stations))
    
    print("\nwarm_start_stations 좌표:")
    print(repr(warm_start_stations))

if __name__ == "__main__":
    main()