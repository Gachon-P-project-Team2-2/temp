"""DQN 기반 기지국 위치 최적화.

현재 optimizer 계약에 맞춘 독립 알고리즘이다. OPEX, 트래픽 예측, 건물 회절
모델은 의도적으로 포함하지 않는다.
"""
from __future__ import annotations

from collections import deque
import logging
import random
import time
from typing import Any

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ..metaheuristics._shared import calculate_score, clip_stations, random_stations

log = logging.getLogger(__name__)


def _load_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "DQN Placement 알고리즘을 사용하려면 PyTorch(torch)가 필요합니다. "
            "`pip install torch` 후 다시 실행하세요."
        ) from exc
    return torch, nn, optim


def _make_model(nn, in_channels: int = 4, action_dim: int = 5):
    class PlacementCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((10, 10)),
            )
            self.head = nn.Sequential(
                nn.Linear(32 * 10 * 10, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.reshape(x.size(0), -1)
            return self.head(x)

    return PlacementCNN()


class DQNPlacementOptimizer(Optimizer):
    name = "DQN Placement"
    hyperparams = [
        HyperParam("episodes", "int", default=20, min=1, max=200, step=5,
                   label="episodes (학습 에피소드 수)"),
        HyperParam("steps_per_episode", "int", default=10, min=1, max=50, step=1,
                   label="steps_per_episode (에피소드당 스텝)"),
        HyperParam("step_size", "float", default=50.0, min=10.0, max=200.0, step=10.0,
                   label="step_size (이동 크기, m)"),
        HyperParam("random_state", "int", default=42, min=-1, max=99999,
                   label="random_state (시드, -1=랜덤)"),
    ]

    def optimize(
        self,
        problem: ProblemInput,
        n_stations: int,
        episodes: int = 20,
        steps_per_episode: int = 10,
        step_size: float = 50.0,
        random_state: int = 42,
        callback=None,
    ) -> OptimizationResult:
        torch, nn, optim = _load_torch()

        if random_state != -1:
            seed = int(random_state)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        t0 = time.perf_counter()
        log.info(
            "DQN Placement start: n_stations=%d episodes=%d steps=%d step=%.1f N=%d",
            n_stations, episodes, steps_per_episode, step_size, len(problem.X),
        )

        grid_h, grid_w = 40, 40
        traffic_grid = self._traffic_grid(problem, grid_h, grid_w)
        feasible_grid = self._feasible_grid(problem, grid_h, grid_w)

        def get_state(stations: np.ndarray, active_idx: int) -> np.ndarray:
            state = np.zeros((4, grid_h, grid_w), dtype=np.float32)
            state[0] = traffic_grid
            state[1] = feasible_grid
            
            from ..base import _resolve_station_params
            radii, _ = _resolve_station_params(problem, n_stations)
            
            dx = problem.width_m / grid_w
            dy = problem.height_m / grid_h
            y_coords = (np.arange(grid_h) + 0.5) * dy
            x_coords = (np.arange(grid_w) + 0.5) * dx
            xx, yy = np.meshgrid(x_coords, y_coords)
            
            for i, (x, y) in enumerate(stations):
                r = radii[i]
                if r <= 0:
                    r = max(dx, dy)
                mask = ((xx - x)**2 + (yy - y)**2) <= r**2
                if i == active_idx:
                    state[3][mask] = 1.0
                else:
                    state[2][mask] = 1.0
            return state

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("DQN Placement device=%s", device)
        model = _make_model(nn).to(device)
        target_model = _make_model(nn).to(device)
        target_model.load_state_dict(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        replay_buffer: deque = deque(maxlen=2000)
        batch_size = 32
        gamma = 0.9
        epsilon = 1.0
        epsilon_decay = 0.95
        epsilon_min = 0.1

        best_stations = random_stations(n_stations, problem)
        best_score = calculate_score(best_stations, problem)
        history = [{"iter": 0, "best_score": best_score, "stations": best_stations.tolist()}]
        if callback is not None:
            callback(0, episodes, best_stations.copy(), best_score)

        total_steps = 0

        for ep in range(1, int(episodes) + 1):
            current_stations = random_stations(n_stations, problem)
            current_score = calculate_score(current_stations, problem)

            for _ in range(int(steps_per_episode)):
                for k in range(n_stations):
                    state = get_state(current_stations, k)
                    if random.random() < epsilon:
                        action = random.randint(0, 4)
                    else:
                        with torch.no_grad():
                            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                        action = int(torch.argmax(q_values).item())

                    next_stations = current_stations.copy()
                    next_stations[k] = self._move_station(next_stations[k], action, step_size)
                    clip_stations(next_stations, problem)

                    next_score = calculate_score(next_stations, problem)
                    reward = next_score - current_score

                    if action != 4 and np.allclose(current_stations[k], next_stations[k]):
                        reward -= 0.1

                    next_state = get_state(next_stations, k)
                    replay_buffer.append((state, action, float(reward), next_state))

                    current_stations = next_stations
                    current_score = next_score

                    if current_score > best_score:
                        best_score = current_score
                        best_stations = current_stations.copy()

                    if len(replay_buffer) >= batch_size:
                        batch = random.sample(replay_buffer, batch_size)
                        states_b, actions_b, rewards_b, next_states_b = zip(*batch)
                        states_t = torch.FloatTensor(np.asarray(states_b)).to(device)
                        actions_t = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                        rewards_t = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
                        next_states_t = torch.FloatTensor(np.asarray(next_states_b)).to(device)

                        current_q = model(states_t).gather(1, actions_t)
                        with torch.no_grad():
                            max_next_q = target_model(next_states_t).max(1)[0].unsqueeze(1)
                            target_q = rewards_t + gamma * max_next_q

                        loss = loss_fn(current_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                total_steps += 1
                if total_steps % 50 == 0:
                    target_model.load_state_dict(model.state_dict())

            history.append({"iter": ep, "best_score": best_score, "stations": best_stations.tolist()})
            if callback is not None:
                callback(ep, episodes, best_stations.copy(), best_score)

        elapsed = time.perf_counter() - t0
        log.info("DQN Placement done: best_score=%.4f elapsed=%.3fs", best_score, elapsed)
        if best_score == 0.0:
            log.warning("DQN Placement score=0: 커버리지가 전혀 없습니다.")

        total_cost = None
        opex_history = None
        if getattr(problem, "score_mode", "traffic") == "total_cost" or getattr(problem, "_force_total_cost", False):
            total_cost = -best_score
            from ..opex_evaluator import evaluate_opex
            _, opex_history = evaluate_opex(best_stations, problem)

        metrics = compute_metrics(best_stations, problem)

        return OptimizationResult(
            stations=best_stations,
            score=best_score,
            metrics=metrics,
            history=history,
        )

    @staticmethod
    def _traffic_grid(problem: ProblemInput, grid_h: int, grid_w: int) -> np.ndarray:
        traffic_series = getattr(problem, "traffic_series", None)
        if traffic_series is not None and getattr(traffic_series, "size", 0) > 0:
            mean_traffic = np.mean(traffic_series, axis=0)
        else:
            mean_traffic = problem.weights

        grid, _, _ = np.histogram2d(
            problem.X[:, 1],
            problem.X[:, 0],
            bins=[grid_h, grid_w],
            range=[[0, problem.height_m], [0, problem.width_m]],
            weights=mean_traffic,
        )
        max_value = float(grid.max()) if grid.size else 0.0
        if max_value > 0:
            grid = grid / max_value
        return grid.astype(np.float32)

    @staticmethod
    def _feasible_grid(problem: ProblemInput, grid_h: int, grid_w: int) -> np.ndarray:
        pool = getattr(problem, "feasible_station_points", None)
        if pool is None:
            pool = getattr(problem, "station_candidate_points", None)
        if pool is None:
            return np.ones((grid_h, grid_w), dtype=np.float32)

        pool = np.asarray(pool, dtype=float)
        if len(pool) == 0:
            return np.zeros((grid_h, grid_w), dtype=np.float32)

        grid, _, _ = np.histogram2d(
            pool[:, 1],
            pool[:, 0],
            bins=[grid_h, grid_w],
            range=[[0, problem.height_m], [0, problem.width_m]],
        )
        return (grid > 0).astype(np.float32)

    @staticmethod
    def _move_station(point: np.ndarray, action: int, step_size: float) -> np.ndarray:
        moved = point.copy()
        if action == 0:
            moved[1] -= step_size
        elif action == 1:
            moved[1] += step_size
        elif action == 2:
            moved[0] -= step_size
        elif action == 3:
            moved[0] += step_size
        return moved
