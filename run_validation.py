import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput, CostParams
from optimizers.kmeans import KMeansOptimizer
from optimizers.drl.dqn_placement import DQNPlacementOptimizer

def main():
    print("Setting up test environment...")
    # Setup exactly like the UI test would
    env = SyntheticEnvironment(width_m=2000, height_m=2000, resolution_m=50)
    env.generate_traffic_pattern("multi_hotspot")
    
    radius = 500.0
    capacity = 100.0  # Limit capacity to force overload
    n_stations = 5
    
    problem = ProblemInput.from_env(
        env,
        radius_m=radius,
        capacity=capacity,
        tx_power_dbm=43.0,
        bandwidth_mhz=10.0,
        score_mode="traffic"
    )
    
    print(f"\n--- MAP INFO ---")
    print(f"Total Traffic to Cover: {np.sum(problem.weights):.2f} Mbps")
    print(f"Capacity per Station: {capacity} Mbps")
    
    # 1. Run KMeans
    print(f"\n--- RUNNING K-MEANS ---")
    km = KMeansOptimizer()
    res_km = km.optimize(problem, n_stations=n_stations, random_state=42)
    
    print("K-Means Metrics:")
    print(f"- Covered Traffic: {res_km.metrics['covered_traffic']:.2f}")
    print(f"- Overload Traffic: {np.sum(np.maximum(0, res_km.metrics['station_loads'] - res_km.metrics['capacity'])):.2f}")
    print(f"- Max Station Load: {np.max(res_km.metrics['station_loads']):.2f}")
    
    # 2. Run DQN
    print(f"\n--- RUNNING DQN PLACEMENT ---")
    dqn = DQNPlacementOptimizer()
    res_dqn = dqn.optimize(problem, n_stations=n_stations, episodes=50, steps_per_episode=10, random_state=42)
    
    diag = res_dqn.metrics["diagnostics"]
    
    # Analyze Logs
    cap_logs = diag["capacity_logs"]
    print("\n[1] Capacity Check:")
    for log in cap_logs:
        print(f"  Ep {log['ep']}: Min={log['min']:.1f}, Max={log['max']:.1f}, Mean={log['mean']:.1f}")
        
    print("\n[2] Overload Penalty Activation:")
    overload_arr = np.array(diag["overload_logs"])
    print(f"  Max Overload: {np.max(overload_arr):.2f}")
    print(f"  Mean Overload: {np.mean(overload_arr):.2f}")
    print(f"  Times Overload > 0: {np.sum(overload_arr > 0)} / {len(overload_arr)}")
    
    print("\n[3] Reward Component Influence (Average over all steps):")
    delta_arr = np.array(diag["delta_cov_logs"])
    max_load_arr = np.array(diag["max_load_logs"])
    
    print(f"  Avg |Delta Covered|: {np.mean(np.abs(delta_arr)):.4f}")
    print(f"  Avg Overload Penalty: {np.mean(overload_arr * 2.0):.4f} (Weight 2.0)")
    print(f"  Avg Max Load Penalty: {np.mean(max_load_arr * 0.05):.4f} (Weight 0.05)")
    
    print("\n[4] DQN Learning Progress (Sample):")
    ep_stats = diag["ep_stats"]
    for i in [0, len(ep_stats)//2, len(ep_stats)-1]:
        st = ep_stats[i]
        print(f"  Ep {st['ep']:3d}: Best Score={st['best_score']:.2f}, Avg Reward={st['avg_reward']:.4f}, Epsilon={st['epsilon']:.3f}")

    print("\n[5] KMeans vs DQN Final Comparison:")
    def format_res(name, r):
        m = r.metrics
        cov = m['covered_traffic']
        ovr = np.sum(np.maximum(0, m['station_loads'] - m['capacity']))
        max_load = np.max(m['station_loads'])
        tp = m['total_throughput_mbps']
        print(f"{name:10s} | Cov Traffic: {cov:7.2f} | Overload: {ovr:7.2f} | Max Load: {max_load:7.2f} | Throughput: {tp:7.2f}")
        
    format_res("K-Means", res_km)
    format_res("DQN", res_dqn)

if __name__ == "__main__":
    main()