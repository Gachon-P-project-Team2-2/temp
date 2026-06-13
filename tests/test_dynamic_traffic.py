from __future__ import annotations

import numpy as np

from app import (
    APP_STATE,
    _LAST_ACCESSED,
    apply_algo_compare_row,
    build_station_circles,
    compute_dynamic_scenario_summary,
    compute_frame_metrics,
    compute_status_overlay,
    env_dataframe_for_current_frame,
    evaluate_operation_optimization,
    live_visualization_state,
    normalize_operation_params,
    operation_active_mask_for_frame,
    operation_comparison_rows,
    render_algo_compare_results,
    render_stats_panel,
)
from environment import SyntheticEnvironment


def _make_env() -> SyntheticEnvironment:
    return SyntheticEnvironment(width_km=1.0, height_km=1.0, resolution_m=100)


def _component_texts(component) -> list[str]:
    if component is None:
        return []
    if isinstance(component, str):
        return [component]
    if isinstance(component, (list, tuple)):
        texts: list[str] = []
        for item in component:
            texts.extend(_component_texts(item))
        return texts
    props = component.to_plotly_json().get("props", {}) if hasattr(component, "to_plotly_json") else {}
    return _component_texts(props.get("children"))


def test_dynamic_traffic_types_create_time_series():
    for dynamic_type in ("fixed_variation", "moving_hotspot", "switching_locations"):
        env = _make_env()
        series = env.generate_dynamic_traffic_pattern_density(
            area_demand_mbps_km2=150,
            pattern="multi_hotspot",
            time_steps=6,
            variation=0.5,
            drift_m=200,
            dynamic_type=dynamic_type,
            params={
                "centers": [(3, 3), (7, 7)],
                "n_centers": 2,
                "sigma_x": 1.5,
                "sigma_y": 1.5,
                "noise_std": 0.0,
            },
        )

        assert series.shape == (6, env.rows, env.cols)
        assert env.traffic_series is not None
        assert np.isfinite(series).all()
        assert series.max() > series.min()


def test_fixed_variation_keeps_peak_location_stable():
    env = _make_env()
    series = env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=5,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(4, 6)],
            "n_centers": 1,
            "sigma_x": 1.2,
            "sigma_y": 1.2,
            "noise_std": 0.0,
        },
    )

    peak_locations = [np.unravel_index(int(np.argmax(frame)), frame.shape) for frame in series]
    assert len(set(peak_locations)) == 1


def test_switching_locations_changes_peak_location():
    env = _make_env()
    series = env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="center_hotspot",
        time_steps=8,
        variation=0.8,
        dynamic_type="switching_locations",
        params={"noise_std": 0.0},
    )

    peak_locations = {np.unravel_index(int(np.argmax(frame)), frame.shape) for frame in series}
    assert len(peak_locations) > 1


def test_frame_metrics_change_with_dynamic_frame():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=5,
        variation=0.6,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_lat, station_lon = env.local_points_to_geo(np.array([[500.0, 500.0]]))[0]
    opt_results = {
        "stations_geo": [{"lat": float(station_lat), "lon": float(station_lon)}],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0],
        },
    }

    frame0 = compute_frame_metrics(env, opt_results, None, frame_index=0)
    frame1 = compute_frame_metrics(env, opt_results, None, frame_index=1)
    summary = compute_dynamic_scenario_summary(env, opt_results, None)

    assert frame0 is not None
    assert frame1 is not None
    assert frame0["total_traffic"] != frame1["total_traffic"]
    assert summary is not None
    assert summary["avg_traffic_coverage_pct"] >= summary["worst_traffic_coverage_pct"]


def test_operation_optimization_evaluates_always_on_policy():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=4,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_lat, station_lon = env.local_points_to_geo(np.array([[500.0, 500.0]]))[0]
    opt_results = {
        "stations_geo": [{"lat": float(station_lat), "lon": float(station_lon)}],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0],
        },
    }

    result = evaluate_operation_optimization(env, opt_results, None, "always-on")

    assert result is not None
    assert result["policy"] == "always-on"
    assert result["frame_count"] == 4
    assert result["total_opex"] > 0
    assert result["baseline"]["policy"] == "always-on"
    assert all(row["active_count"] == 1 for row in result["history"])
    comparison = operation_comparison_rows(result)
    assert comparison
    assert comparison[0]["항목"] == "총 OPEX"
    assert "전(always-on)" in comparison[0]
    assert "후(always-on)" in comparison[0]


def test_operation_optimization_uses_custom_cost_params():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=4,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_lat, station_lon = env.local_points_to_geo(np.array([[500.0, 500.0]]))[0]
    opt_results = {
        "stations_geo": [{"lat": float(station_lat), "lon": float(station_lon)}],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0],
        },
    }

    default_result = evaluate_operation_optimization(env, opt_results, None, "always-on")
    custom_result = evaluate_operation_optimization(
        env,
        opt_results,
        None,
        "always-on",
        {"load_power_w": 0.0},
    )
    low_tx_results = {
        **opt_results,
        "prop_params": {**opt_results["prop_params"], "tx_power_dbm": [33.0]},
    }
    high_tx_result = evaluate_operation_optimization(
        env,
        opt_results,
        None,
        "always-on",
        {"load_power_w": 0.0},
    )
    low_tx_result = evaluate_operation_optimization(
        env,
        low_tx_results,
        None,
        "always-on",
        {"load_power_w": 0.0},
    )

    assert default_result is not None
    assert custom_result is not None
    assert high_tx_result is not None
    assert low_tx_result is not None
    assert custom_result["operation_params"]["load_power_w"] == 0.0
    assert custom_result["total_energy_cost"] < default_result["total_energy_cost"]
    assert low_tx_result["total_energy_cost"] < high_tx_result["total_energy_cost"]


def test_operation_active_mask_updates_map_state_for_frame():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=2,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_geo = env.local_points_to_geo(np.array([[500.0, 500.0], [800.0, 800.0]]))
    opt_results = {
        "stations_geo": [
            {"lat": float(station_geo[0, 0]), "lon": float(station_geo[0, 1])},
            {"lat": float(station_geo[1, 0]), "lon": float(station_geo[1, 1])},
        ],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0, 43.0],
        },
    }
    operation_results = {
        "history": [
            {"active_mask": [True, False]},
            {"active_mask": [False, True]},
        ]
    }
    env.set_traffic_frame(1)

    active_mask = operation_active_mask_for_frame(operation_results, 2, env.dynamic_frame_index)
    assert active_mask is not None
    assert active_mask.tolist() == [False, True]

    _status, overlay_loads, _sinr = compute_status_overlay(
        env,
        env_dataframe_for_current_frame(env),
        opt_results,
        {"n_stations": 2},
        None,
        active_mask=active_mask,
    )
    assert overlay_loads[0] == 0.0

    circles = build_station_circles(
        opt_results,
        {"n_stations": 2},
        None,
        None,
        overlay_loads,
        active_mask=active_mask,
    )
    first_circle_props = circles[0].to_plotly_json()["props"]
    assert first_circle_props["color"] == "#6b7280"
    assert first_circle_props["dashArray"] == "5 5"


def test_compute_frame_metrics_applies_operation_active_mask():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=2,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_geo = env.local_points_to_geo(np.array([[500.0, 500.0], [800.0, 800.0]]))
    opt_results = {
        "stations_geo": [
            {"lat": float(station_geo[0, 0]), "lon": float(station_geo[0, 1])},
            {"lat": float(station_geo[1, 0]), "lon": float(station_geo[1, 1])},
        ],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0, 43.0],
        },
    }

    all_active = compute_frame_metrics(env, opt_results, None, active_mask=np.array([True, True]))
    all_sleep = compute_frame_metrics(env, opt_results, None, active_mask=np.array([False, False]))

    assert all_active is not None
    assert all_sleep is not None
    assert all_active["covered_traffic"] > all_sleep["covered_traffic"]
    assert all_active["total_throughput_mbps"] > all_sleep["total_throughput_mbps"]
    assert all_active["total_tx_power_w"] > all_sleep["total_tx_power_w"]
    assert all_sleep["covered_traffic"] == 0.0
    assert all_sleep["total_throughput_mbps"] == 0.0
    assert all_sleep["total_tx_power_w"] == 0.0
    assert all_sleep["n_stations"] == 0
    assert all_sleep["active_station_count"] == 0
    assert all_sleep["total_station_count"] == 2
    assert all_sleep["operation_active_mask_applied"] is True


def test_stats_panel_reflects_operation_active_mask():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=2,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_geo = env.local_points_to_geo(np.array([[500.0, 500.0], [800.0, 800.0]]))
    opt_results = {
        "stations_geo": [
            {"lat": float(station_geo[0, 0]), "lon": float(station_geo[0, 1])},
            {"lat": float(station_geo[1, 0]), "lon": float(station_geo[1, 1])},
        ],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0, 43.0],
        },
    }
    session_id = "unit-test-stats-operation-mask"
    APP_STATE[session_id] = {
        "env": env,
        "opt_results": opt_results,
        "operation_results": {
            "history": [
                {"active_mask": [False, False]},
                {"active_mask": [False, False]},
            ]
        },
    }

    try:
        cards = render_stats_panel(
            {"version": 1},
            {"version": 1},
            {"version": 1},
            None,
            "전체 동일",
            43.0,
            3.5,
            10.0,
            3.0,
            1,
            session_id,
        )
        texts = _component_texts(cards)
        assert "0 / 2 활성" in texts
        assert "0.0 Mbps" in texts
    finally:
        APP_STATE.pop(session_id, None)
        _LAST_ACCESSED.pop(session_id, None)


def test_live_visualization_state_uses_current_common_and_propagation_controls():
    opt_results = {
        "stations_geo": [
            {"lat": 37.0, "lon": 127.0},
            {"lat": 37.001, "lon": 127.001},
        ],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [30.0, 31.0],
        },
    }
    stale_individual_specs = [
        {"tx_power_dbm": 21.0, "bandwidth_mhz": 5.0},
        {"tx_power_dbm": 22.0, "bandwidth_mhz": 6.0},
    ]

    live_results, live_specs = live_visualization_state(
        opt_results,
        "전체 동일",
        stale_individual_specs,
        ui_tx_power=47.0,
        ui_path_loss_exp=4.2,
        ui_bandwidth_mhz=40.0,
        ui_sinr_threshold=8.0,
        ui_max_coord=3,
    )

    assert live_results is not None
    assert live_specs is not None
    prop = live_results["prop_params"]
    assert prop["path_loss_exponent"] == 4.2
    assert prop["sinr_threshold_db"] == 8.0
    assert prop["bandwidth_mhz"] == 40.0
    assert prop["max_coord_stations"] == 3
    assert prop["tx_power_dbm"] == [47.0, 47.0]
    assert [row["tx_power_dbm"] for row in live_specs] == [47.0, 47.0]
    assert [row["bandwidth_mhz"] for row in live_specs] == [40.0, 40.0]

    individual_results, individual_specs = live_visualization_state(
        opt_results,
        "기지국별 개별",
        stale_individual_specs,
        ui_tx_power=47.0,
        ui_path_loss_exp=4.2,
        ui_bandwidth_mhz=40.0,
        ui_sinr_threshold=8.0,
        ui_max_coord=3,
    )

    assert individual_results is not None
    assert individual_specs == stale_individual_specs
    assert individual_results["prop_params"]["tx_power_dbm"] == [21.0, 22.0]


def test_operation_params_expose_and_clamp_non_auto_settings():
    params = normalize_operation_params(
        {
            "switching_cost": -1.0,
            "uncovered_penalty": 12.0,
            "overload_penalty": 25.0,
            "dqn_epsilon": 1.5,
            "dqn_epsilon_decay": -0.5,
            "dqn_epsilon_min": 2.0,
        }
    )

    assert params["switching_cost"] == 0.0
    assert params["uncovered_penalty"] == 12.0
    assert params["overload_penalty"] == 25.0
    assert params["dqn_epsilon"] == 1.0
    assert params["dqn_epsilon_decay"] == 0.0
    assert params["dqn_epsilon_min"] == 1.0


def test_algo_compare_results_are_selectable_and_apply_selected_row():
    session_id = "unit-test-algo-compare"
    APP_STATE[session_id] = {
        "algo_compare_results": [
            {
                "algo": "Algo A",
                "score": 10.0,
                "covered_traffic": 5.0,
                "coverage_pct": 50.0,
                "area_pct": 40.0,
                "mean_sinr_db": 3.2,
                "total_throughput_mbps": 12.0,
                "elapsed_sec": 0.2,
                "opt_results": {"algo": "Algo A", "score": 10.0},
                "opt_stats": {"n_stations": 1},
            },
            {
                "algo": "Algo B",
                "score": 20.0,
                "covered_traffic": 8.0,
                "coverage_pct": 80.0,
                "area_pct": 70.0,
                "mean_sinr_db": 5.1,
                "total_throughput_mbps": 18.0,
                "elapsed_sec": 0.4,
                "opt_results": {"algo": "Algo B", "score": 20.0},
                "opt_stats": {"n_stations": 2},
            },
        ]
    }

    try:
        rendered = render_algo_compare_results({"version": 1}, session_id)
        table = rendered[1].children[0]

        assert table.id == "algo-compare-datatable"
        assert table.row_selectable == "single"

        meta, _status = apply_algo_compare_row([1], session_id)

        assert "version" in meta
        assert APP_STATE[session_id]["opt_results"]["algo"] == "Algo B"
        assert APP_STATE[session_id]["opt_stats"]["n_stations"] == 2
    finally:
        APP_STATE.pop(session_id, None)
        _LAST_ACCESSED.pop(session_id, None)
