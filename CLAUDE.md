# CLAUDE.md — bs_simulator 작업 지침

이 파일은 Claude Code(자율 에이전트 포함)가 본 리포지토리에서 작업할 때 따라야 할 컨텍스트와 규칙을 정의한다.

## 프로젝트 한 줄 요약
Streamlit 기반 **기지국(Base Station) 위치 최적화 시뮬레이터**. 지도 위에 합성 트래픽/장애물을 만들고, 4가지 최적화 알고리즘으로 기지국을 배치해 커버리지를 시각화한다.

## 아키텍처 (3-파일 구조)
- **`app.py`** — Streamlit UI · 세션 상태 · Folium 지도 렌더링 · 알고리즘 실행 트리거. 모든 사용자 상호작용의 진입점.
- **`environment.py`** — `SyntheticEnvironment` 클래스. 합성 트래픽(가우시안 핫스팟 + base intensity), 장애물 생성(`random/circle/strip/grid/mixed`), Local↔Geo 좌표 변환, 마스킹.
- **`optimizers.py`** — `BaseStationOptimizer` 클래스. K-Means / Random Walk / Simulated Annealing / Tabu Search. `_calculate_score`를 공통 평가 함수로 사용.

## 좌표계 (가장 헷갈리는 부분 — 반드시 숙지)
- **Local 좌표**: 좌상단 원점, 단위 m. `(0,0)` = NW, `(width_m, height_m)` = SE.
  - 단, `get_local_data()`는 좌하단 원점이 아니라 **meshgrid의 y_grid 그대로** 반환한다 (위→아래 증가). `get_local_data_top_left()`는 `y`를 반전한다.
- **Geo 좌표**: 위/경도. 변환 시 `lat_span = height_km / 111.0`, `lon_span = width_km / 88.0` 사용 (서울 위도 기준 근사).
- **`stations_geo` DataFrame은 `[lat, lon]` 순서**, Folium은 `[lat, lon]`을 받지만 GeoJSON 좌표는 `[lon, lat]`이다 — 변환 시 항상 확인.

## 데이터 흐름
1. 사용자가 지도 bounds 확인 → `geodesic`으로 width/height_km 계산
2. `SyntheticEnvironment(...)` 초기화 → `generate_traffic` → `generate_obstacles` → `apply_masking`
3. `BaseStationOptimizer(env, radius, capacity)` → `run_<algo>(k)` → `(centers_local, score)`
4. `get_stats(centers)` → 트래픽/면적 커버리지, 기지국별 부하, 용량 초과 분석
5. `convert_to_geo` → `st.session_state['opt_results']`/`opt_stats`에 저장 → 지도 재렌더

## 점수 함수 (`_calculate_score`)
```
score = sum(min(station_load, capacity)) + 0.1 * covered_grid_count
```
- 트래픽 커버리지가 주, 면적 가중치는 보조 (0.1)
- **용량(capacity) 초과분은 점수에 반영 안 됨** — overload 페널티는 없음
- 점수 함수를 변경하려면 `optimizers.py::_calculate_score`만 수정하면 4개 알고리즘 모두 영향 받는다

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py            # 또는 run_app.bat (Windows)
```
- Python 3.10+ (레포에 3.13 venv 흔적 있음)
- `requirements.txt`의 `sklearn`은 deprecated 별칭 — 실패 시 `scikit-learn`으로 교체

## 작업 시 규칙

### 항상 지킬 것
- **Local↔Geo 변환은 `env.lat_min/max`, `env.lon_min/max`만 사용**. 다른 좌표 소스를 만들지 말 것.
- **`st.session_state` 키 이름 유지**: `env`, `opt_results`, `opt_stats`, `range_results`, `map_view`. UI 다른 부분들이 키 이름에 직접 의존한다.
- **벡터화 유지**: 거리/할당 계산은 numpy broadcasting (`X[:,None,:] - centers[None,:,:]`). 명시적 for 루프로 되돌리지 말 것.
- **알고리즘 추가 시**: `run_<name>(self, n_stations, ...)` 시그니처 유지 → `(best_centers, best_score)` 반환 → `app.py`의 `algo` selectbox와 if-elif 분기에 등록.

### 하지 말 것
- 점수 함수에 임의로 페널티/보너스를 추가하지 말 것 (사용자 합의 필요 — 4개 알고리즘 결과를 한꺼번에 바꾼다).
- `traffic_map`을 직접 in-place 수정하지 말 것 (마스킹 외엔 항상 새 array).
- `requirements.txt`에 무거운 의존성(torch, tensorflow 등) 추가 금지 — 현재는 scikit/scipy/shapely 수준의 경량 스택 유지.
- README.md를 마음대로 재구성하지 말 것 — 한국어 톤과 섹션 순서 보존.

### UI 변경 시
- 사이드바 섹션 번호(1. 환경 설정 → 2. 시각화 → 3. 계산 알고리즘) 순서 유지.
- 한국어/영어 혼용 라벨 패턴 유지 (예: `"커버리지 반경 (m)"`, `"기지국 개수 설정"`).
- 변경 후 반드시 `streamlit run app.py`로 직접 확인. 타입체크/단위테스트만으로는 UI 회귀를 잡을 수 없음.

## 성능 주의점
- `resolution_m`이 작아질수록 `rows*cols`가 제곱으로 늘고, `_calculate_score`의 `(N, K, 2)` broadcasting 메모리도 같이 커진다. K-Means 외 알고리즘은 iterations만큼 반복 호출되므로 큰 그리드+큰 K에서 매우 느려진다.
- 범위 탐색(Range)은 `(k_max - k_min + 1)`번 실행 — 진행률 표시 유지 필수.

## 알려진 한계 / 잠재 개선
- Overload 페널티 없음 → 용량 초과해도 점수 손실 X
- 장애물 패턴 5종으로 고정 — 도시 빌딩/도로망 등 실측 데이터 연동 미구현
- Random Walk / SA / Tabu의 하이퍼파라미터(iterations, step_size)가 UI에 노출되지 않음 — 코드 상수
- `KMeans(random_state=42)` 고정 → 같은 입력에 항상 같은 결과 (재현성↑, 다양성↓)

## 브랜치 & 커밋 규칙
**작업 단위마다 새 브랜치 + 커밋**한다. `main`에 직접 커밋하지 않는다.

- 브랜치 이름: `feat/<짧은-설명>` · `fix/<...>` · `chore/<...>` · `refactor/<...>` · `docs/<...>`
- 한 브랜치 = 한 논리적 작업 단위. 서로 다른 관심사는 별도 브랜치로 분리.
- 커밋 시점: 작업이 자기충족적 상태가 될 때마다 (테스트 가능, 구문 OK). 한 브랜치에 여러 커밋 OK.
- `TODO.md` 갱신은 해당 작업의 마지막 커밋에 함께 포함.
- push/PR은 사용자가 명시적으로 요청한 경우에만. 기본은 로컬 커밋까지.
- 작업 시작 시: `git switch main && git pull` → `git switch -c <type>/<slug>` → 작업 → 커밋.

## 작업 관리 — `TODO.md`
모든 작업은 리포지토리 루트의 **`TODO.md`** 파일로 관리한다. 인메모리 Task 도구 대신 파일을 단일 진실 원본(SoTT)으로 사용해 세션이 끊겨도 진행 상태가 보존되게 한다.

### 규칙
- 새 작업 요청 시: 먼저 `TODO.md`를 읽어 기존 항목과 중복/충돌 여부 확인 → 필요하면 항목 추가.
- 작업 시작 시: 해당 항목을 `- [ ]` → `- [~]` (진행 중)로 바꾸고 `**진행 중:** YYYY-MM-DD` 메모 추가.
- 작업 완료 시: `- [~]` → `- [x]`로 바꾸고 한 줄 결과 메모를 인라인으로 남김 (예: `- [x] SA 하이퍼파라미터 UI 노출 — app.py sidebar expander 추가`).
- 중단/블록: `- [!]` 표시 + `**블록:** <이유>` 메모.
- 삭제 대신 `- [x] ~~취소됨~~` 처리 — 과거 결정의 기록을 남긴다.
- 섹션 구조: `## 진행 중` / `## 대기` / `## 완료` / `## 아이디어`. 위에서 아래로 이동.
- 커밋 시 `TODO.md`도 같이 커밋한다. 코드 변경과 작업 상태가 한 이력에 엮이도록.

### 작업 플로우 (매 요청마다)
1. `TODO.md` 읽기 → 현재 상태 파악
2. 새 요청이면 `대기`에 추가, 기존 건이면 `진행 중`으로 이동
3. 작업 수행
4. 완료 표시 + 결과 메모 → 파일 저장

## 자율 작업 시 체크리스트
1. `TODO.md`를 먼저 열었는가? 중복 작업은 아닌가?
2. 변경 전: 영향 받는 좌표계(Local/Geo)와 세션 상태 키를 확인했는가?
3. 점수/알고리즘 변경: 4개 알고리즘 전부에 영향 가는지 확인했는가?
4. UI 변경: 실제로 `streamlit run`으로 띄워 클릭해봤는가?
5. 변경 후: 합성 데이터 생성 → 알고리즘 실행 → 지도 표시까지 골든 패스가 깨지지 않았는가?
6. `TODO.md`에 결과를 반영하고 저장했는가?
