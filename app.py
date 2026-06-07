import streamlit as st
import numpy as np
import pandas as pd
import io
import json
import folium
import time
from branca.element import MacroElement, Template
from folium import GeoJson
from streamlit_folium import st_folium
from geopy.distance import geodesic
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from environment import SyntheticEnvironment
from obstacle_sources import (
    filter_polygons,
    geojson_to_polygons,
    load_osm_polygons_with_cache,
)
from optimizers import (
    REGISTRY, get_optimizer, ProblemInput, convert_to_geo, HyperParam,
)
from patterns import PATTERN_CHOICES

st.set_page_config(layout="wide", page_title="Simulator")

OSM_OBSTACLE_TYPE_LABELS = ["건물", "수역/물길", "도로"]
OSM_OBSTACLE_TYPE_VALUES = {
    "건물": "building",
    "수역/물길": ("water", "waterway"),
    "도로": "road",
}
OSM_OBJECT_USAGE_MODES = ["장애물로 사용", "기지국 후보로 사용"]

if 'map_view' not in st.session_state:
    st.session_state['map_view'] = {
        'center': [37.4979, 127.0276],
        'zoom': 14
    }


def render_hyperparam_widget(p: HyperParam):
    """HyperParam 스키마에 따라 적절한 Streamlit 위젯 렌더링."""
    label = p.label or p.name
    if p.kind == "int":
        if p.step is not None:
            return st.slider(label, int(p.min), int(p.max), int(p.default), step=int(p.step))
        if p.min is not None and p.max is not None:
            # min/max가 주어지고 step이 없으면 number_input (음수 허용 등)
            return st.number_input(label, int(p.min), int(p.max), int(p.default))
        return st.number_input(label, value=int(p.default))
    elif p.kind == "float":
        fmt = None
        step = p.step if p.step is not None else 0.01
        # cooling_rate 같이 소수점 3자리 필요한 경우 자동 포맷
        if step < 0.01:
            fmt = "%.3f"
        if fmt:
            return st.slider(label, float(p.min), float(p.max), float(p.default),
                             step=float(step), format=fmt)
        return st.slider(label, float(p.min), float(p.max), float(p.default), step=float(step))
    elif p.kind == "choice":
        return st.selectbox(label, p.choices, index=p.choices.index(p.default))
    elif p.kind == "bool":
        return st.checkbox(label, value=bool(p.default))
    raise ValueError(f"Unknown HyperParam.kind: {p.kind}")


def clear_optimization_state():
    for k in ('opt_results', 'opt_stats', 'range_results'):
        if k in st.session_state:
            del st.session_state[k]


def load_map_obstacles(env: SyntheticEnvironment, source: str, uploaded_geojson,
                       min_area_m2: float, max_obstacles: int | None,
                       osm_obstacle_types: list[str] | None = None,
                       osm_object_mode: str = OSM_OBJECT_USAGE_MODES[0]):
    """OSM/GeoJSON 데이터를 로컬 좌표계 객체로 가져온다.

    - 장애물 모드: polygon 목록을 반환
    - 후보 모드: 각 객체의 대표점 좌표 목록을 반환
    """
    if source == "OSM 지도 데이터":
        if not osm_obstacle_types:
            raise ValueError("OSM 오브젝트 종류를 하나 이상 선택해주세요.")
        try:
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
                obstacle_types=osm_obstacle_types,
            )
        except TypeError as exc:
            # 하위/구버전 함수 호환: obstacle_types 인자가 없는 경우 기존 인터페이스로 fallback
            if "unexpected keyword argument 'obstacle_types'" not in str(exc):
                raise
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
            )
    elif source == "GeoJSON 업로드":
        if uploaded_geojson is None:
            raise ValueError("GeoJSON 파일을 먼저 업로드해주세요.")
        geo_polygons = geojson_to_polygons(uploaded_geojson.getvalue())
        raw_count = len(geo_polygons)
    else:
        return [], 0

    local_polygons = []
    for polygon in geo_polygons:
        local_polygons.extend(env.geo_to_local_polygons(polygon))
    if osm_object_mode == "기지국 후보로 사용":
        candidate_points = [
            poly.representative_point().coords[0]
            for poly in local_polygons
            if poly.area > 0
        ]
        return candidate_points, raw_count

    return filter_polygons(local_polygons, min_area_m2, max_obstacles), raw_count


def apply_obstacle_source(env: SyntheticEnvironment, source: str, uploaded_geojson,
                          min_area_m2: float, max_obstacles: int | None,
                          obstacle_pattern: str, num_obstacles: int,
                          osm_obstacle_types: list[str] | None = None,
                          osm_object_mode: str = OSM_OBJECT_USAGE_MODES[0],
                          append: bool = False):
    def apply_candidate_mode(points):
        if append:
            env.append_station_candidate_points(points)
        else:
            env.set_station_candidate_points(points)
        env.obstacles = []
        env.obstacles_geo = []
        env.remask_traffic()

    if source == "합성":
        if osm_object_mode == "기지국 후보로 사용":
            generated = SyntheticEnvironment(
                center_lat=env.center_lat,
                center_lon=env.center_lon,
                width_km=env.width_km,
                height_km=env.height_km,
                resolution_m=env.resolution_m,
            )
            generated.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
            candidate_points = [
                poly.representative_point().coords[0]
                for poly in generated.obstacles
                if poly.area > 0
            ]
            apply_candidate_mode(candidate_points)
            return len(candidate_points), num_obstacles

        if append:
            before = len(env.obstacles)
            generated = SyntheticEnvironment(
                center_lat=env.center_lat,
                center_lon=env.center_lon,
                width_km=env.width_km,
                height_km=env.height_km,
                resolution_m=env.resolution_m,
            )
            generated.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
            env.append_obstacles(generated.obstacles)
            return len(env.obstacles) - before, num_obstacles
        env.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
        env.remask_traffic()
        return len(env.obstacles), num_obstacles

    polygons, raw_count = load_map_obstacles(
        env,
        source,
        uploaded_geojson,
        min_area_m2,
        max_obstacles,
        osm_obstacle_types=osm_obstacle_types,
        osm_object_mode=osm_object_mode,
    )
    if osm_object_mode == "기지국 후보로 사용":
        apply_candidate_mode(polygons)
        return len(env.station_candidate_points), raw_count

    if append:
        env.append_obstacles(polygons)
    else:
        env.replace_obstacles(polygons)
        env.clear_station_candidate_points()
    return len(polygons), raw_count


def add_dynamic_traffic_playback(map_obj, env: SyntheticEnvironment, df: pd.DataFrame,
                                 interval_s: float, traffic_series: np.ndarray | None = None):
    """Folium 지도 안에서 GeoJSON layer style만 갱신하는 JS 재생 컨트롤 추가."""
    cell_indices = df.index.to_numpy(dtype=int)
    if traffic_series is None:
        flat_series = env.traffic_series.reshape(env.traffic_series.shape[0], -1)
    else:
        flat_series = traffic_series.reshape(traffic_series.shape[0], -1)
    frame_values = np.rint(flat_series[:, cell_indices]).astype(int).tolist()
    frames_json = json.dumps(frame_values, separators=(",", ":"))
    map_name = map_obj.get_name()
    control_id = f"{map_name}_traffic_playback"
    initial_frame = int(getattr(env, 'dynamic_frame_index', 0))
    interval_ms = max(100, int(float(interval_s) * 1000))

    js = f"""
    (function() {{
        var frames = {frames_json};
        var controlId = "{control_id}";
        var currentFrame = {initial_frame};
        var timer = null;
        var cells = [];

        function opacityForTraffic(value) {{
            return Math.max(0.05, Math.min(value / 150.0, 0.85));
        }}

        function updateFrame(frameIndex) {{
            if (!frames.length || !cells.length) {{
                return;
            }}
            currentFrame = (frameIndex + frames.length) % frames.length;
            var frame = frames[currentFrame];
            for (var i = 0; i < cells.length; i++) {{
                var value = frame[i] || 0;
                var isObstacle = false;
                if (cells[i].feature && cells[i].feature.properties &&
                        Object.prototype.hasOwnProperty.call(cells[i].feature.properties, "is_obstacle")) {{
                    isObstacle = Boolean(cells[i].feature.properties.is_obstacle);
                }}
                if (isObstacle) {{
                    if (cells[i].setStyle) {{
                        cells[i].setStyle({{
                            fillColor: "#808080",
                            fillOpacity: 0.65
                        }});
                    }}
                    if (cells[i].feature && cells[i].feature.properties) {{
                        cells[i].feature.properties.traffic = 0;
                        cells[i].feature.properties.status = "Obstacle";
                    }}
                    continue;
                }}
                if (cells[i].setStyle) {{
                    cells[i].setStyle({{
                        fillColor: "#ff0000",
                        fillOpacity: opacityForTraffic(value)
                    }});
                }}
                if (cells[i].feature && cells[i].feature.properties) {{
                    cells[i].feature.properties.traffic = value;
                    cells[i].feature.properties.status = "Playback";
                }}
            }}
            var frameLabel = document.getElementById(controlId + "-frame");
            if (frameLabel) {{
                frameLabel.textContent = "Traffic Frame " + currentFrame + " / " + (frames.length - 1);
            }}
        }}

        function findLeafletMap() {{
            if (typeof L === "undefined") {{
                return null;
            }}
            var keys = Object.keys(window);
            for (var i = 0; i < keys.length; i++) {{
                try {{
                    if (window[keys[i]] instanceof L.Map) {{
                        return window[keys[i]];
                    }}
                }} catch (e) {{}}
            }}
            return null;
        }}

        function findTrafficLayer() {{
            if (typeof L === "undefined") {{
                return null;
            }}
            var keys = Object.keys(window);
            for (var i = 0; i < keys.length; i++) {{
                var candidate = window[keys[i]];
                try {{
                    if (!(candidate instanceof L.LayerGroup) || typeof candidate.eachLayer !== "function") {{
                        continue;
                    }}
                    var hasTrafficFeature = false;
                    candidate.eachLayer(function(layer) {{
                        if (layer.feature && layer.feature.properties &&
                                Object.prototype.hasOwnProperty.call(layer.feature.properties, "traffic")) {{
                            hasTrafficFeature = true;
                        }}
                    }});
                    if (hasTrafficFeature) {{
                        return candidate;
                    }}
                }} catch (e) {{}}
            }}
            return null;
        }}

        function initPlayback(attempt) {{
            var map = findLeafletMap();
            var trafficLayer = findTrafficLayer();
            if (!map || !trafficLayer) {{
                if (attempt < 50) {{
                    window.setTimeout(function() {{
                        initPlayback(attempt + 1);
                    }}, 100);
                }}
                return;
            }}

            trafficLayer.eachLayer(function(layer) {{
                cells.push(layer);
            }});

            var TrafficPlaybackControl = L.Control.extend({{
                options: {{position: "topleft"}},
                onAdd: function() {{
                    var container = L.DomUtil.create("div", "traffic-playback-control");
                    container.id = controlId;
                    container.style.cssText = [
                        "background: rgba(20,20,20,0.88)",
                        "color: white",
                        "padding: 8px 10px",
                        "border-radius: 4px",
                        "box-shadow: 0 1px 5px rgba(0,0,0,0.45)",
                        "font-size: 13px",
                        "line-height: 1.35"
                    ].join(";");
                    container.innerHTML =
                        '<div id="' + controlId + '-frame" style="font-weight: 700; margin-bottom: 6px;">' +
                        'Traffic Frame ' + currentFrame + ' / ' + (frames.length - 1) +
                        '</div>' +
                        '<button id="' + controlId + '-play" type="button" style="margin-right: 4px;">지도 재생</button>' +
                        '<button id="' + controlId + '-stop" type="button">정지</button>';
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);
                    return container;
                }}
            }});

            map.addControl(new TrafficPlaybackControl());

            document.getElementById(controlId + "-play").onclick = function() {{
                if (timer !== null) {{
                    return;
                }}
                timer = window.setInterval(function() {{
                    updateFrame(currentFrame + 1);
                }}, {interval_ms});
            }};

            document.getElementById(controlId + "-stop").onclick = function() {{
                if (timer !== null) {{
                    window.clearInterval(timer);
                    timer = null;
                }}
            }};

            updateFrame(currentFrame);
        }}

        initPlayback(0);
    }})();
    """
    playback_element = MacroElement()
    playback_element._name = "DynamicTrafficPlayback"
    playback_element._template = Template(
        "{% macro script(this, kwargs) %}" + js + "{% endmacro %}"
    )
    map_obj.add_child(playback_element)


with st.sidebar:
    st.title("시뮬레이터 제어")

    st.header("1. 환경 설정")
    resolution_m = st.number_input("격자 크기 (m)", 50, 500, 100, step=10)

    with st.expander("트래픽 세부 설정"):
        traffic_pattern = st.selectbox(
            "트래픽 패턴",
            PATTERN_CHOICES,
            index=0,
            help="multi_hotspot=기존(레거시, m 단위 spread). 나머지는 bs_opt 포팅 패턴.",
        )
        base_intensity = st.slider("기초 트래픽량 (base)", 0, 50, 10)
        max_intensity = st.slider("최대 트래픽량 (max)", 50, 500, 100, step=10,
                                   help="정규화된 패턴에 곱해지는 스케일 (multi_hotspot 제외)")

        # 레거시 multi_hotspot은 m 단위 spread_m을 받음 (기존 파라미터 유지)
        if traffic_pattern == "multi_hotspot":
            num_hotspots = st.slider("핫스팟 개수", 1, 10, 5)
            spread_m = st.slider("핫스팟 확산 반경 (m)", 100, 1000, 300, step=50)
        else:
            num_hotspots = 5  # 사용 안 함
            spread_m = 300

        dynamic_traffic = st.checkbox("동적 트래픽 생성", value=False)
        dynamic_time_steps = 12
        dynamic_variation = 0.25
        dynamic_drift_m = 300
        if dynamic_traffic:
            dynamic_time_steps = st.slider("시간 단계 수", 2, 48, 12)
            dynamic_variation = st.slider("시간 변화 강도", 0.0, 1.0, 0.25, step=0.05)
            dynamic_drift_m = st.slider("공간 이동 범위 (m)", 0, 2000, 300, step=50)

    with st.expander("오브젝트 세부 설정"):
        osm_object_mode = OSM_OBJECT_USAGE_MODES[0]
        osm_object_mode = st.radio(
            "오브젝트 사용 방식",
            OSM_OBJECT_USAGE_MODES,
            horizontal=True,
        )

        obstacle_source = st.selectbox(
            "오브젝트 소스",
            ["합성", "OSM 지도 데이터", "GeoJSON 업로드"],
            index=0,
        )
        obstacle_pattern = "mixed"
        num_obstacles = 3
        min_obstacle_area_m2 = 0.0
        max_map_obstacles = None
        uploaded_geojson = None
        osm_obstacle_types = None
        
        

        if obstacle_source == "합성":
            obstacle_pattern = st.selectbox("오브젝트 생성 패턴", ["mixed", "random", "circle", "strip", "grid"], index=0)
            num_obstacles = st.slider("오브젝트 개수", 0, 10, 3)
        elif obstacle_source == "OSM 지도 데이터":
            st.write("OSM 오브젝트 타입")
            selected_osm_types = []
            for osm_type in OSM_OBSTACLE_TYPE_LABELS:
                if st.checkbox(
                    osm_type,
                    value=True,
                    key=f"osm_object_type_{osm_type.replace('/', '_')}",
                ):
                    selected_osm_types.append(osm_type)
            if not selected_osm_types:
                st.warning("최소 하나 이상의 OSM 오브젝트 타입을 선택해야 합니다.")
            osm_obstacle_types = []
            for osm_type in selected_osm_types:
                value = OSM_OBSTACLE_TYPE_VALUES[osm_type]
                if isinstance(value, tuple):
                    osm_obstacle_types.extend(value)
                else:
                    osm_obstacle_types.append(value)
            # 수역/물길은 하나의 체크박스로 묶어도 내부적으로는 물 경계 요소를 함께 조회한다.
            osm_obstacle_types = list(dict.fromkeys(osm_obstacle_types))
        else:
            uploaded_geojson = st.file_uploader("오브젝트 GeoJSON", type=["json", "geojson"])
            if osm_object_mode == "장애물로 사용":
                min_obstacle_area_m2 = st.slider("최소 오브젝트 면적 (m²)", 0, 5000, 100, step=100)
                max_map_obstacles = st.slider("최대 오브젝트 개수", 1, 500, 100, step=10)

    create_clicked = st.button("가상 데이터 생성", type="primary")

    if 'env' in st.session_state:
        st.markdown("---")
        env = st.session_state['env']

        if getattr(env, 'traffic_series', None) is not None and env.traffic_series.shape[0] > 1:
            st.header("동적 트래픽")
            current_frame = int(getattr(env, 'dynamic_frame_index', 0))
            selected_frame = st.slider(
                "트래픽 시간 프레임",
                0,
                env.traffic_series.shape[0] - 1,
                current_frame,
            )
            if selected_frame != current_frame:
                env.set_traffic_frame(selected_frame)
                st.session_state['env'] = env
                clear_optimization_state()
                current_frame = selected_frame

            st.session_state['traffic_js_interval'] = st.slider(
                "브라우저 재생 간격 (초)",
                0.2,
                2.0,
                float(st.session_state.get('traffic_js_interval', 0.5)),
                step=0.1,
            )

            st.caption(f"현재 지도 프레임: {current_frame} / {env.traffic_series.shape[0] - 1}")
            st.caption("재생/정지는 지도 왼쪽 위 컨트롤에서 실행됩니다.")

            st.markdown("---")

        with st.expander("지도 오브젝트 관리"):
            candidate_count = 0 if env.station_candidate_points is None else len(env.station_candidate_points)
            st.caption(f"현재 오브젝트: {len(env.obstacles)}개")
            if candidate_count:
                st.caption(f"현재 기지국 후보: {candidate_count}개")
            if 'obstacle_status' in st.session_state:
                st.caption(st.session_state['obstacle_status'])
            obstacle_apply_mode = st.radio(
                "오브젝트 적용 방식",
                ["교체", "추가"],
                horizontal=True,
            )
            apply_obstacles_clicked = st.button("오브젝트 다시 불러오기/적용")
            clear_obstacles_clicked = st.button("오브젝트 초기화")

            if apply_obstacles_clicked:
                apply_status = st.empty()
                apply_progress = st.progress(0)
                apply_status.info("오브젝트를 적용하고 있습니다...")
                apply_progress.progress(20)
                try:
                    applied_count, raw_count = apply_obstacle_source(
                        env,
                        obstacle_source,
                        uploaded_geojson,
                        float(min_obstacle_area_m2),
                        max_map_obstacles,
                        obstacle_pattern,
                        int(num_obstacles),
                        osm_obstacle_types=osm_obstacle_types,
                        osm_object_mode=osm_object_mode,
                        append=obstacle_apply_mode == "추가",
                    )
                    apply_progress.progress(80)
                    st.session_state['env'] = env
                    applied_type = "기지국 후보" if osm_object_mode == "기지국 후보로 사용" else "장애물"
                    st.session_state['obstacle_status'] = (
                        f"{obstacle_source}({applied_type}): 원본 {raw_count}개 중 {applied_count}개 적용"
                    )
                    apply_status.success(f"{applied_type} 적용 완료")
                    apply_progress.progress(100)
                    clear_optimization_state()
                    st.rerun()
                except Exception as exc:
                    apply_status.error(f"적용 실패: {exc}")
                    st.error(f"적용 실패: {exc}")
                finally:
                    apply_progress.empty()

            if clear_obstacles_clicked:
                clear_status = st.empty()
                clear_status.info("지도를 초기화하고 있습니다...")
                env.clear_obstacles()
                clear_status.success("지도 요소 초기화 완료")
                st.session_state['env'] = env
                st.session_state['obstacle_status'] = "지도 요소(오브젝트/후보) 초기화 완료"
                clear_optimization_state()
                st.rerun()

            st.markdown("---")

        st.header("데이터 내보내기")
        df_geo = env.get_dataframe()
        st.download_button("GIS 데이터 (CSV)", df_geo.to_csv(index=False).encode('utf-8'), "traffic_geo.csv", "text/csv")
        local_data = env.get_local_data_top_left()
        df_local = pd.DataFrame(local_data, columns=['x', 'y', 'traffic'])
        st.download_button("Local 데이터 (CSV)", df_local.to_csv(index=False).encode('utf-8'), "traffic_local.csv", "text/csv")
        buffer = io.BytesIO()
        np.save(buffer, env.traffic_map)
        st.download_button("Map 데이터 (NPY)", buffer, "traffic_map.npy", "application/octet-stream")
        if getattr(env, 'traffic_series', None) is not None:
            show_series_download = st.checkbox("전체 시계열 다운로드 표시", value=False)
            if show_series_download:
                series_buffer = io.BytesIO()
                np.save(series_buffer, env.traffic_series)
                st.download_button(
                    "Time Series 데이터 (NPY)",
                    series_buffer,
                    "traffic_series.npy",
                    "application/octet-stream",
                )

    st.markdown("---")
    st.header("2. 시각화 설정")
    map_layer_mode = st.radio("지도 표시 모드", ["트래픽 분포 (Traffic)", "커버리지 상태 (Status)"], index=1)

    st.markdown("---")
    st.header("3. 계산 알고리즘")

    with st.expander("기지국 스펙 (Spec)", expanded=True):
        radius_m = st.slider("커버리지 반경 (m)", 100, 10000, 300, step=50)
        capacity = st.number_input("최대 용량 (Traffic)", 500, 1000000000, 2000, step=100)

    available = [cls.name for cls in REGISTRY]
    algo = st.selectbox("알고리즘 선택", available)
    optimizer = get_optimizer(algo)

    hyperparams = {}
    if optimizer.hyperparams:
        with st.expander("하이퍼파라미터", expanded=False):
            for p in optimizer.hyperparams:
                hyperparams[p.name] = render_hyperparam_widget(p)

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

# 수렴 그래프 (단일 k & history 존재 시)
if ('opt_results' in st.session_state
        and opt_mode == "고정 개수 (Fixed)"
        and st.session_state['opt_results'].get('history')):
    hist = st.session_state['opt_results']['history']
    with st.expander("수렴 이력 (Convergence)", expanded=False):
        df_hist = pd.DataFrame(hist)
        fig, ax = plt.subplots(figsize=(10, 3))
        if 'best_score' in df_hist.columns:
            ax.plot(df_hist['iter'], df_hist['best_score'], label='Best Score', color='tab:green')
        if 'current_score' in df_hist.columns:
            ax.plot(df_hist['iter'], df_hist['current_score'], label='Current Score',
                    color='tab:blue', alpha=0.4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# 기지국 위치 변화 시각화 (history에 stations 스냅샷이 있을 때)
if ('opt_results' in st.session_state
        and opt_mode == "고정 개수 (Fixed)"
        and st.session_state['opt_results'].get('history')
        and 'env' in st.session_state):
    hist = st.session_state['opt_results']['history']
    snapshots = [h for h in hist if 'stations' in h]
    if len(snapshots) >= 2:
        with st.expander("기지국 위치 변화 (Station Movement)", expanded=False):
            env = st.session_state['env']
            n_snaps = len(snapshots)
            _problem = ProblemInput.from_env(
                env,
                radius_m=radius_m,
                capacity=capacity,
                station_candidate_points=env.station_candidate_points,
            )

            # geo 변환된 트래픽 히트맵 extent
            _geo_extent = [env.lon_min, env.lon_max, env.lat_min, env.lat_max]

            # 모든 스냅샷의 geo 좌표를 미리 계산
            _all_geo = [convert_to_geo(s['stations'], _problem) for s in snapshots]

            def _draw_snap(ax, snap, snap_i, init_geo):
                """스냅샷 하나를 geo 좌표로 차트에 그리기."""
                ax.imshow(np.flipud(env.traffic_map), extent=_geo_extent,
                          cmap='YlOrRd', alpha=0.5, aspect='auto')
                st_geo = _all_geo[snap_i]

                # 이동 궤적 (처음 ~ 현재 스냅샷)
                if snap_i >= 1:
                    for si in range(len(st_geo)):
                        trail_lon = [_all_geo[j][si, 1] for j in range(snap_i + 1)]
                        trail_lat = [_all_geo[j][si, 0] for j in range(snap_i + 1)]
                        ax.plot(trail_lon, trail_lat, '-', color='blue', alpha=0.4, lw=1.2, zorder=3)

                # 초기 위치 — 회색 ×
                ax.scatter(init_geo[:, 1], init_geo[:, 0], c='gray', marker='x',
                           s=80, linewidths=1.5, zorder=4)
                # 현재 위치 — 빨간 ●
                ax.scatter(st_geo[:, 1], st_geo[:, 0], c='red', marker='o', s=100,
                           edgecolors='white', linewidths=1.5, zorder=5)

                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f"Iteration {snap['iter']}  |  Score: {snap['best_score']:.1f}")

            _init_geo = convert_to_geo(snapshots[0]['stations'], _problem)

            # 슬라이더 + 재생 버튼
            snap_idx = st.slider(
                "스냅샷 선택", 0, n_snaps - 1, n_snaps - 1,
                key="snap_slider",
            )
            if st.button("▶ 애니메이션 재생"):
                chart_slot = st.empty()
                progress_slot = st.empty()
                for frame in range(n_snaps):
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    _draw_snap(ax2, snapshots[frame], frame, _init_geo)
                    chart_slot.pyplot(fig2)
                    plt.close(fig2)
                    progress_slot.progress(frame / max(n_snaps - 1, 1))
                    time.sleep(0.3)
                progress_slot.empty()
            else:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                _draw_snap(ax2, snapshots[snap_idx], snap_idx, _init_geo)
                st.pyplot(fig2)

# 지도 표시
m = folium.Map(
    location=st.session_state['map_view']['center'],
    zoom_start=st.session_state['map_view']['zoom'],
    control_scale=True,
    tiles="CartoDB dark_matter"
)

if 'env' in st.session_state:
    env = st.session_state['env']
    raw_series = env.get_raw_traffic_series()
    if raw_series is not None:
        flat_traffic = raw_series[env.dynamic_frame_index].ravel()
    else:
        flat_traffic = env.get_raw_traffic_map().ravel()
    obstacle_mask = env.get_obstacle_mask().ravel()
    df = pd.DataFrame({
        "lat": env.lat_grid.ravel(),
        "lon": env.lon_grid.ravel(),
        "traffic": flat_traffic,
        "is_obstacle": obstacle_mask,
    })

    status_list = []

    if 'opt_results' in st.session_state:
        res = st.session_state['opt_results']
        stations = res['stations_geo']

        calc_radius = res.get('radius', 300)
        calc_capacity = st.session_state['opt_stats'].get('capacity', 1000)

        non_obstacle_mask = (~df['is_obstacle']) & (df['traffic'] > 0.1)
        grid_points = df.loc[non_obstacle_mask, ['lat', 'lon', 'traffic']].values
        grid_indices = np.where(non_obstacle_mask.to_numpy())[0]
        station_points = stations[['lat', 'lon']].values

        if len(station_points) > 0:
            grid_status = np.zeros(len(df), dtype=int)
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
                        grid_status[grid_indices[idx]] = 1  # Covered
                    else:
                        grid_status[grid_indices[idx]] = 2  # Overloaded

            grid_status[df['is_obstacle']] = -1

            status_list = grid_status
        else:
            status_list = np.zeros(len(df), dtype=int)
            status_list[df['is_obstacle']] = -1
    else:
        status_list = np.zeros(len(df), dtype=int)
        status_list[df['is_obstacle']] = -1

    lat_step = (env.lat_max - env.lat_min) / env.rows
    lon_step = (env.lon_max - env.lon_min) / env.cols

    features = []
    for idx, (i, row) in enumerate(df.iterrows()):
        r_lat, r_lon, val, is_obstacle = row['lat'], row['lon'], row['traffic'], row['is_obstacle']

        color = '#ff0000'
        opacity = min(val / 150.0, 0.8)
        status_text = "Obstacle" if is_obstacle else "N/A"

        if is_obstacle:
            color = '#808080'
            opacity = 0.65

        if map_layer_mode == "커버리지 상태 (Status)" and len(status_list) > 0 and idx < len(status_list):
            status = status_list[idx]
            if is_obstacle:
                color = '#808080'
                opacity = 0.65
                status_text = "Obstacle"
            elif status == 1:
                color = '#0000ff'  # Blue
            elif status == 2:
                color = '#ffa500'  # Orange
            else:
                color = '#ff0000'  # Red

            opacity = min(val / 150.0 + 0.2, 0.9)
            status_text = {0: "Uncovered", 1: "Covered", 2: "Overloaded", -1: "Obstacle"}.get(status, "N/A")

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
                "is_obstacle": bool(is_obstacle),
                "status": status_text,
                "fillColor": color,
                "fillOpacity": opacity
            }
        }
        features.append(feature)

    if features:
        geojson_data = {"type": "FeatureCollection", "features": features}
        traffic_geojson = GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': x['properties']['fillColor'],
                'color': 'none',
                'weight': 0,
                'fillOpacity': x['properties']['fillOpacity']
            },
            tooltip=folium.GeoJsonTooltip(fields=['traffic', 'status'], aliases=['Traffic:', 'Status:'], style="font-size: 14px; font-weight: bold;")
        )
        traffic_geojson.add_to(m)
        if getattr(env, 'traffic_series', None) is not None and env.traffic_series.shape[0] > 1:
            add_dynamic_traffic_playback(
                m,
                env,
                df,
                st.session_state.get('traffic_js_interval', 0.5),
                raw_series,
            )

    candidate_points = env.local_points_to_geo(env.station_candidate_points)
    if len(candidate_points) > 0:
        candidate_group = folium.FeatureGroup(name="Station Candidates")
        for idx, (lat, lon) in enumerate(candidate_points):
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup=f"Station Candidate #{idx + 1}",
            ).add_to(candidate_group)
        candidate_group.add_to(m)

# 기지국 마커 표시
if 'opt_results' in st.session_state and 'opt_stats' in st.session_state:
    res = st.session_state['opt_results']
    stats = st.session_state['opt_stats']
    stations = res['stations_geo']
    radius = res.get('radius', 300)

    loads = stats.get('station_effective_loads', [])
    capacity_val = stats.get('capacity', 1000)

    for i, (lat, lon) in enumerate(stations.values):
        load = loads[i] if i < len(loads) else 0
        usage_pct = (load / capacity_val) * 100 if capacity_val > 0 else 0

        popup_html = f"""<div style="width:150px"><b>Station #{i+1}</b><br>Load: {int(load)} / {int(capacity_val)}<br>Usage: {usage_pct:.1f}%</div>"""

        icon_color = 'green'
        if usage_pct > 90:
            icon_color = 'red'
        elif usage_pct > 70:
            icon_color = 'orange'

        folium.Marker(location=[lat, lon], popup=folium.Popup(popup_html, max_width=200), icon=folium.Icon(color=icon_color, icon='wifi', prefix='fa')).add_to(m)
        folium.Circle(location=[lat, lon], radius=radius, color=icon_color, fill=True, fill_opacity=0.1).add_to(m)

output = st_folium(m, width="100%", height=700)

if create_clicked:
    create_status = st.empty()
    create_progress = st.progress(0)
    with st.spinner("가상 환경을 생성하고 있습니다."):
        create_progress.progress(10)
        create_status.info("지도를 기준으로 환경 영역을 계산하는 중입니다.")
        if output and output.get('bounds') and output.get('center'):
            bounds = output['bounds']
            sw = (bounds['_southWest']['lat'], bounds['_southWest']['lng'])
            ne = (bounds['_northEast']['lat'], bounds['_northEast']['lng'])
            center = output['center']
            center_lat = center['lat']
            center_lon = center['lng']
            zoom = output.get('zoom', st.session_state['map_view']['zoom'])

            width_km = geodesic((sw[0], sw[1]), (sw[0], ne[1])).km
            height_km = geodesic((sw[0], sw[1]), (ne[0], sw[1])).km
        else:
            center_lat, center_lon = st.session_state['map_view']['center']
            zoom = st.session_state['map_view']['zoom']
            width_km = 2.0
            height_km = 2.0

        st.session_state['map_view'] = {'center': [center_lat, center_lon], 'zoom': zoom}
        create_progress.progress(25)

        create_status.info("트래픽 분포를 생성하고 있습니다.")
        env = SyntheticEnvironment(
            center_lat=center_lat,
            center_lon=center_lon,
            width_km=width_km,
            height_km=height_km,
            resolution_m=resolution_m
        )
        if dynamic_traffic:
            pattern_params = {}
            if traffic_pattern == "multi_hotspot":
                sigma_cells = max(spread_m / max(resolution_m, 1), 1.0)
                pattern_params = {
                    "n_centers": num_hotspots,
                    "sigma_x": sigma_cells,
                    "sigma_y": sigma_cells,
                }
            env.generate_dynamic_traffic_pattern(
                pattern=traffic_pattern,
                time_steps=dynamic_time_steps,
                max_intensity=max_intensity,
                base_intensity=base_intensity,
                variation=dynamic_variation,
                drift_m=dynamic_drift_m,
                params=pattern_params,
            )
        elif traffic_pattern == "multi_hotspot":
            # 레거시 경로: m 단위 spread, max_intensity=100 고정
            env.generate_traffic(num_hotspots=num_hotspots, spread_m=spread_m,
                                 base_intensity=base_intensity,
                                 max_intensity=max_intensity)
        else:
            env.generate_traffic_pattern(
                pattern=traffic_pattern,
                max_intensity=max_intensity,
                base_intensity=base_intensity,
            )
        create_progress.progress(55)

        create_status.info("오브젝트 소스를 적용하는 중입니다.")
        try:
            applied_count, raw_count = apply_obstacle_source(
                env,
                obstacle_source,
                uploaded_geojson,
                float(min_obstacle_area_m2),
                max_map_obstacles,
                obstacle_pattern,
                int(num_obstacles),
                osm_obstacle_types=osm_obstacle_types,
                osm_object_mode=osm_object_mode,
                append=False,
            )
            applied_type = "기지국 후보" if osm_object_mode == "기지국 후보로 사용" else "장애물"
            st.session_state['obstacle_status'] = (
                f"{obstacle_source}({applied_type}): 원본 {raw_count}개 중 {applied_count}개 적용"
            )
            create_progress.progress(90)
            create_status.success("가상 환경 생성 완료")
        except Exception as exc:
            env.clear_obstacles()
            st.session_state['obstacle_status'] = f"적용 실패: {exc}"
            create_status.error(f"적용 실패: {exc}")
        create_progress.progress(100)

        st.session_state['env'] = env
        clear_optimization_state()
        st.rerun()
    create_progress.empty()

if optimize_clicked:
    start_time = time.time()
    if 'env' in st.session_state:
        optimize_status = st.empty()
        env = st.session_state['env']
        problem = ProblemInput.from_env(
            env,
            radius_m=radius_m,
            capacity=capacity,
            station_candidate_points=env.station_candidate_points,
        )

        for k in ('opt_results', 'opt_stats', 'range_results'):
            if k in st.session_state:
                del st.session_state[k]

        if opt_mode == "고정 개수 (Fixed)":
            k_list = [n_stations]
        else:
            k_list = list(range(k_min, k_max + 1))

        range_results = []
        progress_bar = st.progress(0)

        for idx, k in enumerate(k_list):
            optimize_status.info(f"기지국 {k}개 계산 중... ({idx + 1}/{len(k_list)})")
            result = optimizer.optimize(problem, n_stations=k, **hyperparams)

            stations_geo = convert_to_geo(result.stations, problem)
            stations_df = pd.DataFrame(stations_geo, columns=['lat', 'lon'])

            stats_out = dict(result.metrics)
            stats_out['n_stations'] = k

            res_pack = {
                'k': k,
                'score': result.score,
                'covered_traffic': result.metrics['covered_traffic'],
                'covered_area': result.metrics['covered_area'],
                'opt_results': {
                    'algo': algo,
                    'score': result.score,
                    'stations_geo': stations_df,
                    'radius': radius_m,
                    'history': result.history,
                },
                'stats': stats_out,
            }
            range_results.append(res_pack)
            progress_bar.progress((idx + 1) / len(k_list))

        progress_bar.empty()
        optimize_status.success(f"계산 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        st.session_state['range_results'] = range_results

        best_res = max(range_results, key=lambda x: x['score'])
        st.session_state['opt_results'] = best_res['opt_results']
        st.session_state['opt_stats'] = best_res['stats']

        optimize_status.success("최적화 결과 저장 완료")
        st.rerun()
    else:
        st.error("먼저 데이터를 생성해주세요.")
