import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput
from optimizers.metaheuristics._shared import calculate_score
from optimizers.base import compute_metrics
from app import compute_dynamic_scenario_summary

import logging
logging.basicConfig(level=logging.ERROR)

def main():
    print("=== 최대 부하(Max Load) 일치 여부 검증 ===")
    
    # 1. 환경 설정 (동적 트래픽 생성)
    env = SyntheticEnvironment(width_km=2.0, height_km=2.0, resolution_m=50)
    env.generate_traffic_pattern("multi_hotspot") # 24 프레임의 raw_traffic_series 생성
    
    # 2. 임의의 기지국 배치 생성
    n_stations = 5
    np.random.seed(42)
    x = np.random.uniform(0, env.width_m, n_stations)
    y = np.random.uniform(0, env.height_m, n_stations)
    stations_local = np.column_stack([x, y])
    
    # 3. ProblemInput 생성 (DQN이 사용하는 입력)
    problem = ProblemInput.from_env(
        env,
        radius_m=500.0,
        capacity=1e10,
        tx_power_dbm=43.0,
        bandwidth_mhz=10.0,
        score_mode="traffic"
    )
    
    print(f"총 트래픽 (ProblemInput): {np.sum(problem.weights):.2f} Mbps\n")
    
    # --- A. DQN 학습 내부에서 측정되는 방식 (단일 프레임/정적) ---
    dqn_metrics = compute_metrics(stations_local, problem)
    dqn_max_load = np.max(dqn_metrics["station_loads"]) if len(dqn_metrics["station_loads"]) > 0 else 0.0
    
    print("[1] DQN(compute_metrics) 측정 방식")
    print(f"  - 계산된 station_loads 배열: {dqn_metrics['station_loads']}")
    print(f"  - 도출된 max_station_load: {dqn_max_load:.2f} Mbps\n")
    
    # --- B. UI에 표시되는 방식 (동적 24프레임 누적) ---
    # UI는 app.py의 compute_dynamic_scenario_summary()를 호출함
    opt_results = {
        "stations_geo": [{} for _ in range(n_stations)], # 껍데기만 줌
        "stations": stations_local # 테스트용으로 원본 좌표 주입 (실제 UI 구조와 맞춰줌)
    }
    
    # compute_dynamic_scenario_summary 함수 분석
    # 내부적으로 0~23 프레임(T) 전체를 순회하면서
    # 각 프레임별로 compute_frame_metrics 를 호출하여 loads를 구한 뒤
    # 그 중 '가장 큰 값(max_station_load = max(max_station_load, np.max(loads)))'을 찾음.
    
    ui_summary = compute_dynamic_scenario_summary(env, opt_results, station_specs=None)
    
    print("[2] UI 화면(compute_dynamic_scenario_summary) 표시 방식")
    if ui_summary:
        print(f"  - 도출된 max_station_load: {ui_summary['max_station_load']:.2f} Mbps\n")
    else:
        print("  - 동적 트래픽 결과 없음\n")
        
    print("=== 분석 결론 ===")
    print("DQN 내부의 값은 '현재 시점(t=0 또는 정지된 맵)' 단일 프레임에서의 부하량입니다.")
    print("UI에 표시되는 값은 '24시간(전체 프레임) 중 어느 한 시간대라도 치솟았던 최댓값(전역 최댓값)'입니다.")
    print("따라서 트래픽이 움직이는 동적 맵 환경에서는 두 값이 다르게 나올 확률이 매우 높습니다.")

if __name__ == "__main__":
    main()