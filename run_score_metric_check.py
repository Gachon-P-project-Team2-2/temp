import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput, compute_metrics
from optimizers.metaheuristics._shared import calculate_score

import logging
logging.getLogger("optimizers").setLevel(logging.CRITICAL)

def main():
    env = SyntheticEnvironment(width_km=2.0, height_km=2.0, resolution_m=50)
    rng = np.random.default_rng(42)
    env.generate_traffic_pattern_density(area_demand_mbps_km2=100.0, pattern="multi_hotspot", rng=rng)
    
    problem = ProblemInput.from_env(
        env, radius_m=500.0, capacity=1e10, tx_power_dbm=43.0,
        bandwidth_mhz=10.0, score_mode="traffic"
    )
    
    # 이전 단계에서 출력된 best_stations 좌표 사용
    best_stations = np.array([
        [1531.72995008,  814.01079185],
        [ 423.50755567, 1630.36029635],
        [ 394.93719293,  761.26998412],
        [1059.70582925, 1613.93791707],
        [1629.03391461, 1546.09478197]
    ])
    
    score_val = calculate_score(best_stations, problem)
    metrics_val = compute_metrics(best_stations, problem)["covered_traffic"]
    
    print(f"calculate_score(best_stations): {score_val:.4f}")
    print(f"compute_metrics(best_stations)[\"covered_traffic\"]: {metrics_val:.4f}")
    print(f"두 값이 완전히 동일한가?: {score_val == metrics_val}")

if __name__ == "__main__":
    main()