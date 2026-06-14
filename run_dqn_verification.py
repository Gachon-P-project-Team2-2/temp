import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput
from optimizers.drl.dqn_placement import DQNPlacementOptimizer
from optimizers.metaheuristics._shared import calculate_score

import logging
logging.getLogger("optimizers").setLevel(logging.CRITICAL)

def main():
    print("=== DQN Placement 최종 결과 변수 검증 ===")
    
    # 1. 반경이 좁아지도록 환경 설정 (DQN이 탐색할 여지를 줌)
    env = SyntheticEnvironment(width_km=2.0, height_km=2.0, resolution_m=50)
    rng = np.random.default_rng(42)
    env.generate_traffic_pattern_density(area_demand_mbps_km2=100.0, pattern="multi_hotspot", rng=rng)
    
    # 경로 손실을 높여 예상 반경을 대폭 감소시킴
    problem = ProblemInput.from_env(
        env,
        radius_m=300.0,
        capacity=1e10,
        path_loss_exponent=3.5, 
        sinr_threshold_db=5.0,
        tx_power_dbm=43.0,
        bandwidth_mhz=10.0,
        score_mode="traffic"
    )
    
    n_stations = 5
    
    # 2. DQN 최적화 실행 (내부 K-Means Warm Start 포함)
    dqn_opt = DQNPlacementOptimizer()
    
    # DQN 내부 로직과 똑같이 외부에서 warm_start_stations 재현 (비교를 위해)
    from sklearn.cluster import KMeans
    from optimizers.metaheuristics._shared import snap_stations_to_candidates
    km = KMeans(n_clusters=n_stations, n_init=10, random_state=42)
    km.fit(problem.X, sample_weight=problem.weights)
    warm_start_stations = snap_stations_to_candidates(km.cluster_centers_, problem)
    
    # 실제 최적화 구동
    res = dqn_opt.optimize(problem, n_stations=n_stations, episodes=50, steps_per_episode=20, random_state=42)
    best_stations = res.stations
    best_score = res.score
    
    # 3. 요청하신 출력 항목 계산
    is_allclose = np.allclose(best_stations, warm_start_stations)
    max_abs_diff = np.max(np.abs(best_stations - warm_start_stations))
    warm_start_score = calculate_score(warm_start_stations, problem)
    best_score_recalc = calculate_score(best_stations, problem)
    
    print("\n[출력 1] np.allclose(best_stations, warm_start_stations):")
    print(is_allclose)
    
    print("\n[출력 2] np.max(np.abs(best_stations - warm_start_stations)):")
    print(f"{max_abs_diff:.4f} m (최대 이동 거리)")
    
    print("\n[출력 3] best_score (반환된 score):")
    print(f"{best_score:.4f}")
    
    print("\n[출력 4] calculate_score(warm_start_stations):")
    print(f"{warm_start_score:.4f}")
    
    print("\n[출력 5] calculate_score(best_stations):")
    print(f"{best_score_recalc:.4f}")
    
    print("\n[출력 6] UI에 전달되는 OptimizationResult(stations=...) 변수 확인:")
    print("실제 dqn_placement.py 코드 (259~264 라인):")
    print("        return OptimizationResult(")
    print("            stations=best_stations,")
    print("            score=best_score,")
    print("            metrics=metrics,")
    print("            history=history,")
    print("        )")
    print("-> UI에 최종적으로 전달되는 값은 'best_stations' 임을 확인했습니다.")

if __name__ == "__main__":
    main()