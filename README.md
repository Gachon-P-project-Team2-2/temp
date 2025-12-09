BS Simulator (Base Station Placement)
=====================================
Streamlit 기반의 기지국(셀) 배치·커버리지 시뮬레이터입니다. 지도에서 가상(합성) 트래픽과 장애물을 생성하고, 여러 최적화 알고리즘으로 기지국 위치를 계산한 뒤 결과를 지도에 표시합니다.

빠른 시작
---------
1) 의존성 설치
- Python 3.10+ (레포에는 3.13 venv 포함), `pip install -r requirements.txt`

2) 실행
- `streamlit run app.py` (또는 `run_app.bat`)

3) 기본 흐름
- 지도에서 영역을 보며 사이드바에서 **환경 설정 → 가상 데이터 생성 → 계산 알고리즘 실행** 순서로 진행
- 결과는 지도(트래픽/커버리지 레이어)와 지표(커버된 트래픽, 면적, 기지국 수)로 확인

파일 구조 & 책임
-----------------
- `app.py`  
  - Streamlit UI, 상태 관리, 지도 렌더링, 다운로드 버튼, 알고리즘 실행 트리거
- `environment.py`  
  - `SyntheticEnvironment` 클래스: 합성 트래픽/장애물 생성, 마스킹, Geo/Local 데이터 변환
- `optimizers.py`  
  - `BaseStationOptimizer` 클래스: K-Means, Random Walk, Simulated Annealing, Tabu Search로 기지국 위치 탐색, 스코어·통계 계산
- `run_app.bat`  
  - Windows에서 가상환경 활성화 후 Streamlit 실행 스크립트

UI 입력 값(변수) 설명
---------------------
### 1) 환경 / 합성 데이터
- `resolution_m` (격자 크기, m): 셀 단위 크기. 작을수록 더 촘촘한 그리드와 큰 행렬(`rows x cols`) 생성.
- 트래픽
  - `base_intensity`: 전체 영역에 깔리는 기본 트래픽 크기.
  - `num_hotspots`: 핫스팟 개수.
  - `spread_m`: 핫스팟 확산 반경(가우시안 분포 표준편차에 해당).
- 장애물
  - `obstacle_pattern`: `mixed | random | circle | strip | grid`
  - `num_obstacles`: 생성할 장애물 개수.

### 2) 시각화
- `map_layer_mode`: `트래픽 분포(Traffic)` 또는 `커버리지 상태(Status)`
  - Status는 기지국 결과가 있을 때만 동작하며, 각 그리드가 `Uncovered/ Covered / Overloaded` 중 하나로 칠해짐.

### 3) 계산 알고리즘
- `radius_m`: 기지국 커버리지 반경(m).
- `capacity`: 기지국 1개가 처리할 수 있는 최대 트래픽.
- `algo`: `K-Means | Random Walk | Simulated Annealing | Tabu Search`
- `opt_mode`: 기지국 개수 모드
  - 고정: `n_stations`
  - 범위 탐색: `k_min`, `k_max` (범위 내 각 k를 돌며 스코어가 가장 큰 결과를 선택)

합성 데이터(가상 데이터) 생성 & 값 전달 구조
---------------------------------------
1) **지도 범위 입력**: Streamlit 지도에서 보이는 현재 bounds/center를 읽어 영역 크기(km)를 계산  
2) `SyntheticEnvironment` 초기화  
   - 지역 중심(`center_lat`, `center_lon`), 가로/세로 길이(km), `resolution_m` → 내부 Local 좌표계를 만듦  
   - Local 좌표: 좌상단(0,0) 기준, 단위 m  
   - Geo 좌표: 위/경도 범위를 균등 분할
3) `generate_traffic`  
   - 기본 트래픽 + `num_hotspots`개의 2D 가우시안(확산 `spread_m`)을 합성  
   - 결과: `traffic_map` (형태: `[rows, cols]`, 단일 채널 = **트래픽 강도**)
4) `generate_obstacles` + `_convert_obstacles_to_geo`  
   - Shapely 다각형/원/스트립/그리드 폴리곤 생성 후 Geo 좌표로 변환  
5) `apply_masking`  
   - 장애물 내부 그리드의 트래픽을 0으로 설정  
6) 지도 전달  
   - Geo 그리드 DF: `lat, lon, traffic` → Folium GeoJson으로 칠함  
   - 장애물 폴리곤: 회색 폴리곤 오버레이  
   - 다운로드: `traffic_geo.csv`, `traffic_local.csv`, `traffic_map.npy`

채널/데이터 포맷 요약
---------------------
- 트래픽 채널: 1개 (`traffic_map`의 실수 값; 단위는 상대적 스케일)  
- Geo CSV (`traffic_geo.csv`): `lat, lon, traffic`  
- Local CSV (`traffic_local.csv`): `x, y, traffic` (x: 좌→우, y: 위→아래)  
- NPY (`traffic_map.npy`): `rows x cols` float array, 장애물 마스킹 반영  
- 커버리지 상태는 별도 채널이 아니라, 계산 후 렌더링 시 `status` 속성으로 제공 (`Uncovered=0`, `Covered=1`, `Overloaded=2`)

계산(최적화) 파이프라인
----------------------
1) `BaseStationOptimizer(env, radius, capacity)` 초기화 시 Local 데이터 로드  
   - `X`: 좌표 `[x, y]`, `weights`: 트래픽 값  
2) 알고리즘 실행 (예: K-Means) → `centers`(Local)와 `score` 반환  
3) `get_stats(centers)`  
   - 반경 내 커버 여부, 기지국별 부하, 커버된 트래픽/면적 계산  
4) `convert_to_geo(centers)`  
   - Local → Geo 변환하여 지도에 마커·원으로 표시  
5) Streamlit 세션 상태 저장  
   - `opt_results`: `{algo, score, stations_geo(DataFrame), radius}`  
   - `opt_stats`: `{total_traffic, covered_traffic, total_area, covered_area, station_loads, station_effective_loads, capacity, n_stations}`
6) 지도 표시  
   - 트래픽 레이어: `traffic` 값으로 색상/투명도  
   - Status 레이어: `status` 코드로 색상(적/청/주황), 툴팁에 `Traffic`, `Status`

커스터마이징 가이드
-------------------
- 합성 트래픽 강도/노이즈: `SyntheticEnvironment.generate_traffic`의 `max_intensity`, 노이즈 분포를 변경
- 장애물 패턴 추가: `environment.py::_create_single_obstacle`에 새 패턴 분기 추가 후 `pattern` 옵션 확장
- 스코어 함수 조정: `BaseStationOptimizer._calculate_score`에서 효율식 변경 (예: 커버 면적 가중치, 오버로드 페널티 추가)
- 커버리지 반경/용량을 결과에 저장: `app.py`에서 계산 시 사용한 `radius_m`, `capacity`를 `opt_results`/`opt_stats`에 함께 넣어 후속 시각화에 활용 가능