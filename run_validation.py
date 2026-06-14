import os
import sys
import numpy as np

sys.path.insert(0, "/Users/gominjung/Desktop/bs_simulator")

from environment import SyntheticEnvironment
from optimizers.base import ProblemInput
from optimizers.drl.dqn_placement import DQNPlacementOptimizer

import logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Setting up test environment...")
    env = SyntheticEnvironment(width_km=2.0, height_km=2.0, resolution_m=50)
    env.generate_traffic_pattern("multi_hotspot")
    
    radius = 500.0
    capacity = 100.0
    n_stations = 5
    
    problem = ProblemInput.from_env(
        env,
        radius_m=radius,
        capacity=capacity,
        tx_power_dbm=43.0,
        bandwidth_mhz=10.0,
        score_mode="traffic"
    )
    
    print("\n--- RUNNING DQN PLACEMENT (Logging Only) ---")
    dqn = DQNPlacementOptimizer()
    dqn.optimize(problem, n_stations=n_stations, episodes=50, steps_per_episode=10, random_state=42)

if __name__ == "__main__":
    main()