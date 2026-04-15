import streamlit as st
import numpy as np
import pandas as pd
import io
import folium
import time
from folium import GeoJson
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt

from environment import SyntheticEnvironment 
from optimizers import BaseStationOptimizer

st.set_page_config(layout="wide", page_title="Simulator")

if 'map_view' not in st.session_state:
    st.session_state['map_view'] = {
        'center': [37.4979, 127.0276], 
        'zoom': 14
    }

with st.sidebar:
    st.title("시뮬레이터 제어")
    
    st.header("1. 환경 설정")
    resolution_m = st.number_input("격자 크기 (m)", 50, 500, 100, step=10)
    
    with st.expander("트래픽 세부 설정"):
        base_intensity = st.slider("기초 트래픽량", 0, 50, 10)
        num_hotspots = st.slider("핫스팟 개수", 1, 10, 5)
        spread_m = st.slider("핫스팟 확산 반경 (m)", 100, 1000, 300, step=50)
        
    with st.expander("장애물 세부 설정"):
        obstacle_pattern = st.selectbox("장애물 패턴", ["mixed", "random", "circle", "strip", "grid"], index=0)
        num_obstacles = st.slider("장애물 개수", 0, 10, 3)
    
    create_clicked = st.button("가상 데이터 생성", type="primary")
    
    if 'env' in st.session_state:
        st.markdown("---")
        st.header("데이터 내보내기")
        env = st.session_state['env']
        df_geo = env.get_dataframe()
        st.download_button("GIS 데이터 (CSV)", df_geo.to_csv(index=False).encode('utf-8'), "traffic_geo.csv", "text/csv")
        local_data = env.get_local_data_top_left()
        df_local = pd.DataFrame(local_data, columns=['x', 'y', 'traffic'])
        st.download_button("Local 데이터 (CSV)", df_local.to_csv(index=False).encode('utf-8'), "traffic_local.csv", "text/csv")
        buffer = io.BytesIO()
        np.save(buffer, env.traffic_map)
        st.download_button("Map 데이터 (NPY)", buffer, "traffic_map.npy", "application/octet-stream")
    
    st.markdown("---")
    st.header("2. 시각화 설정")
    map_layer_mode = st.radio("지도 표시 모드", ["트래픽 분포 (Traffic)", "커버리지 상태 (Status)"], index=1)
    
    st.markdown("---")
    st.header("3. 계산 알고리즘")
    
    with st.expander("기지국 스펙 (Spec)", expanded=True):
        radius_m = st.slider("커버리지 반경 (m)", 100, 10000, 300, step=50)
        capacity = st.number_input("최대 용량 (Traffic)", 500, 1000000000, 2000, step=100)
    
    algo = st.selectbox("알고리즘 선택", ["K-Means", "Random Walk", "Simulated Annealing", "Tabu Search"])

    hyperparams = {}
    with st.expander("하이퍼파라미터", expanded=False):
        if algo == "K-Means":
            hyperparams['n_init'] = st.slider("n_init (초기화 횟수)", 1, 50, 10)
            hyperparams['random_state'] = st.number_input("random_state (시드, -1=랜덤)", -1, 99999, 42)
        elif algo == "Random Walk":
            hyperparams['iterations'] = st.slider("iterations (반복 수)", 100, 10000, 1000, step=100)
            hyperparams['step_size'] = st.slider("step_size (이동 크기, m)", 1.0, 500.0, 50.0, step=1.0)
        elif algo == "Simulated Annealing":
            hyperparams['iterations'] = st.slider("iterations (반복 수)", 100, 10000, 1000, step=100)
            hyperparams['initial_temp'] = st.slider("initial_temp (초기 온도)", 1.0, 1000.0, 100.0, step=1.0)
            hyperparams['cooling_rate'] = st.slider("cooling_rate (냉각률)", 0.80, 0.999, 0.99, step=0.001, format="%.3f")
            hyperparams['step_size'] = st.slider("step_size (이동 크기, m)", 1.0, 500.0, 50.0, step=1.0)
        elif algo == "Tabu Search":
            hyperparams['iterations'] = st.slider("iterations (반복 수)", 50, 5000, 500, step=50)
            hyperparams['step_size'] = st.slider("step_size (이동 크기, m)", 1.0, 500.0, 50.0, step=1.0)
            hyperparams['tabu_tenure'] = st.slider("tabu_tenure (금기 기간)", 1, 100, 10)

    opt_mode = st.radio("기지국 개수 설정", ["고정 개수 (Fixed)", "범위 탐색 (Range)"])
    
    if opt_mode == "고정 개수 (Fixed)":
        n_stations = st.slider("기지국 수", 1, 100, 5)
        k_min, k_max = n_stations, n_stations
    else:
        c1, c2 = st.columns(2)
        k_min = c1.number_input("최소 개수", 1, 100, 3)
        k_max = c2.number_input("최대 개수", k_min, 200, 10)
        n_stations = k_min 
    
    optimize_clicked = st.button("계산 실행")

st.title("기지국 위치 최적화 시뮬레이터")

# 통계 정보 표시
if 'opt_stats' in st.session_state:
    stats = st.session_state['opt_stats']
    total_t = stats.get('total_traffic', 0)
    cov_t = stats.get('covered_traffic', 0)
    total_a = stats.get('total_area', 0)
    cov_a = stats.get('covered_area', 0)
    
    traffic_cov_pct = (cov_t / total_t) * 100 if total_t > 0 else 0
    area_cov_pct = (cov_a / total_a) * 100 if total_a > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 트래픽", f"{int(total_t)}")
    c2.metric("커버된 트래픽", f"{int(cov_t)} ({traffic_cov_pct:.1f}%)")
    c3.metric("커버된 면적", f"{int(cov_a)} 격자 ({area_cov_pct:.1f}%)")
    c4.metric("기지국 수", f"{stats.get('n_stations', '-')}")
    st.markdown("---")

# 범위 탐색 결과 그래프
if 'range_results' in st.session_state and opt_mode == "범위 탐색 (Range)":
    results = st.session_state['range_results']
    st.subheader("보고서")
    
    df_res = pd.DataFrame(results)
    
    tab1, tab2 = st.tabs(["그래프", "보고서"])
    with tab1:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        color = 'tab:blue'
        ax1.set_xlabel('Number of Stations')
        ax1.set_ylabel('Covered Traffic', color=color)
        ax1.plot(df_res['k'], df_res['covered_traffic'], color=color, marker='o', label='Traffic')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Score (Efficiency)', color=color)  
        ax2.plot(df_res['k'], df_res['score'], color=color, linestyle='--', marker='x', label='Score')
        ax2.tick_params(axis='y', labelcolor=color)
        st.pyplot(fig)
        
        selected_k = st.selectbox("기지국 개수 선택", df_res['k'])
        if st.button("업데이트"):
            selected_res = next(item for item in results if item['k'] == selected_k)
            st.session_state['opt_results'] = selected_res['opt_results']
            st.session_state['opt_stats'] = selected_res['stats']
            st.rerun()

    with tab2:
        st.dataframe(df_res)
    st.markdown("---")

# 지도 표시
m = folium.Map(
    location=st.session_state['map_view']['center'],
    zoom_start=st.session_state['map_view']['zoom'],
    control_scale=True,
    tiles="CartoDB dark_matter"
)

if 'env' in st.session_state:
    env = st.session_state['env']
    df = env.get_dataframe()
    
    status_list = []
    
    if 'opt_results' in st.session_state:
        res = st.session_state['opt_results']
        stations = res['stations_geo']
        
        # 지도에 표시될 반경 값 (저장된 값이 아니라 현재 슬라이더 값 반영? -> 아님, 결과값이어야 함)
        # 하지만 사용자가 "계산 실행"을 누르기 전에는 이전 결과를 보여주는게 맞음
        calc_radius = res.get('radius', 300)
        calc_capacity = st.session_state['opt_stats'].get('capacity', 1000)
        
        grid_points = df[['lat', 'lon', 'traffic']].values
        station_points = stations[['lat', 'lon']].values
        
        if len(station_points) > 0:
            grid_status = np.zeros(len(grid_points), dtype=int) 
            station_allocations = [[] for _ in range(len(station_points))]
            
            x_scale = env.width_m / (env.lon_max - env.lon_min)
            y_scale = env.height_m / (env.lat_max - env.lat_min)
            
            st_x = (station_points[:, 1] - env.lon_min) * x_scale
            st_y = (station_points[:, 0] - env.lat_min) * y_scale
            st_local = np.column_stack((st_x, st_y))
            
            gd_x = (grid_points[:, 1] - env.lon_min) * x_scale
            gd_y = (grid_points[:, 0] - env.lat_min) * y_scale
            gd_local = np.column_stack((gd_x, gd_y))
            
            diff = gd_local[:, np.newaxis, :] - st_local[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            radius_sq = calc_radius ** 2
            
            dist_sq_masked = np.where(dist_sq <= radius_sq, dist_sq, np.inf)
            nearest_idx = np.argmin(dist_sq_masked, axis=1)
            min_dist_sq = np.min(dist_sq_masked, axis=1)
            
            for i in range(len(grid_points)):
                if min_dist_sq[i] != np.inf:
                    s_idx = nearest_idx[i]
                    station_allocations[s_idx].append((i, min_dist_sq[i], grid_points[i, 2]))
            
            for s_idx, allocs in enumerate(station_allocations):
                allocs.sort(key=lambda x: x[1])
                current_load = 0
                for idx, dist, traffic in allocs:
                    if current_load + traffic <= calc_capacity:
                        current_load += traffic
                        grid_status[idx] = 1 # Covered
                    else:
                        grid_status[idx] = 2 # Overloaded
                        
            status_list = grid_status
        else:
            status_list = np.zeros(len(grid_points), dtype=int)
    else:
        status_list = np.zeros(len(df), dtype=int)

    lat_step = (env.lat_max - env.lat_min) / env.rows
    lon_step = (env.lon_max - env.lon_min) / env.cols
    
    features = []
    # enumerate를 사용하여 순차 인덱스 사용 (df.iterrows()의 인덱스는 필터링 후에도 원본 인덱스를 유지할 수 있음)
    for idx, (i, row) in enumerate(df.iterrows()):
        r_lat, r_lon, val = row['lat'], row['lon'], row['traffic']
        
        color = '#ff0000' 
        opacity = min(val / 150.0, 0.8)
        status_text = "N/A"
        
        # 모드에 따른 색상 변경
        if map_layer_mode == "커버리지 상태 (Status)" and len(status_list) > 0 and idx < len(status_list):
            status = status_list[idx]  # enumerate로 얻은 순차 인덱스 사용
            if status == 1: color = '#0000ff' # Blue
            elif status == 2: color = '#ffa500' # Orange
            else: color = '#ff0000' # Red
            
            opacity = min(val / 150.0 + 0.2, 0.9)
            status_text = {0: "Uncovered", 1: "Covered", 2: "Overloaded"}[status]
            
        min_lat, max_lat = r_lat - lat_step/2, r_lat + lat_step/2
        min_lon, max_lon = r_lon - lon_step/2, r_lon + lon_step/2
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            },
            "properties": {
                "traffic": round(val, 2),
                "status": status_text,
                "fillColor": color,
                "fillOpacity": opacity
            }
        }
        features.append(feature)
        
    if features:
        geojson_data = {"type": "FeatureCollection", "features": features}
        GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': x['properties']['fillColor'],
                'color': 'none', 
                'weight': 0,
                'fillOpacity': x['properties']['fillOpacity']
            },
            tooltip=folium.GeoJsonTooltip(fields=['traffic', 'status'], aliases=['Traffic:', 'Status:'], style="font-size: 14px; font-weight: bold;")
        ).add_to(m)

    for poly in env.obstacles_geo:
        coords = [(y, x) for x, y in list(poly.exterior.coords)]
        folium.Polygon(locations=coords, color='gray', fill=True, fill_color='gray', fill_opacity=0.5, popup="Obstacle").add_to(m)

# 기지국 마커 표시
if 'opt_results' in st.session_state and 'opt_stats' in st.session_state:
    res = st.session_state['opt_results']
    stats = st.session_state['opt_stats']
    stations = res['stations_geo']
    radius = res.get('radius', 300) 
    
    loads = stats.get('station_effective_loads', [])
    capacity = stats.get('capacity', 1000)
    
    for i, (lat, lon) in enumerate(stations.values):
        load = loads[i] if i < len(loads) else 0
        usage_pct = (load / capacity) * 100 if capacity > 0 else 0
        
        popup_html = f"""<div style="width:150px"><b>Station #{i+1}</b><br>Load: {int(load)} / {int(capacity)}<br>Usage: {usage_pct:.1f}%</div>"""
        
        icon_color = 'green'
        if usage_pct > 90: icon_color = 'red'
        elif usage_pct > 70: icon_color = 'orange'
        
        folium.Marker(location=[lat, lon], popup=folium.Popup(popup_html, max_width=200), icon=folium.Icon(color=icon_color, icon='wifi', prefix='fa')).add_to(m)
        folium.Circle(location=[lat, lon], radius=radius, color=icon_color, fill=True, fill_opacity=0.1).add_to(m)

output = st_folium(m, width="100%", height=700)

if create_clicked:
    if output and output.get('bounds') and output.get('center'):
        bounds = output['bounds']
        sw = (bounds['_southWest']['lat'], bounds['_southWest']['lng'])
        ne = (bounds['_northEast']['lat'], bounds['_northEast']['lng'])
        center = output['center']
        zoom = output['zoom']
        
        width_km = geodesic((sw[0], sw[1]), (sw[0], ne[1])).km
        height_km = geodesic((sw[0], sw[1]), (ne[0], sw[1])).km
        
        st.session_state['map_view'] = {'center': [center['lat'], center['lng']], 'zoom': zoom}
        
        env = SyntheticEnvironment(
            center_lat=center['lat'],
            center_lon=center['lng'],
            width_km=width_km,
            height_km=height_km,
            resolution_m=resolution_m
        )
        env.generate_traffic(num_hotspots=num_hotspots, spread_m=spread_m, base_intensity=base_intensity)
        env.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
        env.apply_masking()
        
        st.session_state['env'] = env
        if 'opt_results' in st.session_state: del st.session_state['opt_results']
        if 'opt_stats' in st.session_state: del st.session_state['opt_stats']
        if 'range_results' in st.session_state: del st.session_state['range_results']
        st.rerun()

if optimize_clicked:
    start_time = time.time()
    if 'env' in st.session_state:
        env = st.session_state['env']
        optimizer = BaseStationOptimizer(env, radius=radius_m, capacity=capacity)
        
        # 기존 결과 초기화 (갱신이 왜 안될까 -> 그래서 강제로 초기화해버림)
        if 'opt_results' in st.session_state: del st.session_state['opt_results']
        if 'opt_stats' in st.session_state: del st.session_state['opt_stats']
        if 'range_results' in st.session_state: del st.session_state['range_results']
        
        if opt_mode == "고정 개수 (Fixed)":
            k_list = [n_stations]
        else:
            k_list = range(k_min, k_max + 1)
        
        range_results = []
        progress_bar = st.progress(0)
        
        for idx, k in enumerate(k_list):
            if algo == "K-Means": centers, score = optimizer.run_kmeans(k, **hyperparams)
            elif algo == "Random Walk": centers, score = optimizer.run_random_walk(k, **hyperparams)
            elif algo == "Simulated Annealing": centers, score = optimizer.run_simulated_annealing(k, **hyperparams)
            elif algo == "Tabu Search": centers, score = optimizer.run_tabu_search(k, **hyperparams)
            
            stats = optimizer.get_stats(centers)
            stats['n_stations'] = k 
            
            centers_geo = optimizer.convert_to_geo(centers)
            stations_df = pd.DataFrame(centers_geo, columns=['lat', 'lon'])
            
            res_pack = {
                'k': k,
                'score': score,
                'covered_traffic': stats['covered_traffic'],
                'covered_area': stats['covered_area'],
                'opt_results': {'algo': algo, 'score': score, 'stations_geo': stations_df, 'radius': radius_m},
                'stats': stats
            }
            range_results.append(res_pack)
            progress_bar.progress((idx + 1) / len(k_list))
            
        progress_bar.empty()
        
        st.session_state['range_results'] = range_results
        
        best_res = max(range_results, key=lambda x: x['score'])
        st.session_state['opt_results'] = best_res['opt_results']
        st.session_state['opt_stats'] = best_res['stats']
        
        st.success("계산 완료 (소요 시간: {:.2f}초)".format(time.time() - start_time))
        st.rerun()
    else:
        st.error("먼저 데이터를 생성해주세요.")
