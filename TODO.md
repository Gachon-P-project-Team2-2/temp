# TODO

작업 관리 파일. 규칙은 `CLAUDE.md`의 "작업 관리 — `TODO.md`" 섹션 참조.

상태 표기: `[ ]` 대기 · `[~]` 진행 중 · `[x]` 완료 · `[!]` 블록

## 진행 중
- [~] **[bs_opt 통합 Phase 1]** Genetic Algorithm 포팅 — 다음 브랜치

## 대기
- [ ] **[bs_opt 통합]** 트래픽 패턴 8종 포팅 — `bs_opt/kmj/core/grid.py::generate_synthetic_traffic`의 패턴을 `SyntheticEnvironment`에 흡수, UI selectbox 노출
- [ ] **[Phase 3]** RL 스캐폴딩 — `optimizers/rl/` (gym env, RewardShaper, Observation). 실제 알고리즘은 별도 브랜치.
- [ ] Overload 페널티 옵션 추가 — 점수 함수에 toggle/가중치, UI 노출
- [ ] KMeans 외 알고리즘의 시드 옵션화 (재현성)
- [ ] 실측 장애물 데이터(OSM 빌딩 폴리곤 등) 연동 검토
- [ ] Playwright MCP로 UI 회귀 테스트 자동화 (MCP 이제 사용 가능)

## 완료
- [x] **[Phase 2]** Optimizer as Plugin 아키텍처 마이그레이션 (2026-04-15) — `optimizers/` 패키지, base.py/kmeans.py/metaheuristics/, HyperParam 스키마 기반 UI 자동 생성, Playwright 회귀 테스트 통과
- [x] 아키텍처 결정 문서화 (ADR) (2026-04-15) — `docs/architecture_decisions.md`. "Optimizer as Plugin" 구조 채택.
- [x] bs_opt 통합 가능성 검토 (2026-04-15) — 보고서 `docs/bs_opt_integration_review.md`, 3-phase 로드맵 제시
- [x] 리포지토리 분석 및 `CLAUDE.md` 작성 (2026-04-15)
- [x] SA/Tabu/RandomWalk/KMeans 하이퍼파라미터 사이드바 expander 노출 (2026-04-15) — `app.py`에 algo별 동적 슬라이더, `**hyperparams`로 전달
- [x] `requirements.txt`의 `sklearn` → `scikit-learn` 교체 (2026-04-15) — deprecated 별칭 대응
- [x] KMeans `random_state` 파라미터화 (`-1`=시드 미고정) (2026-04-15)
- [x] Node.js LTS 설치 + Playwright MCP 등록 (2026-04-15) — winget으로 설치, `~/.claude.json`에 등록됨. 현재 세션에서는 MCP 도구가 로드되지 않아 다음 세션부터 사용 가능.

## 아이디어
- SA/Tabu 수렴 곡선 실시간 차트 (iteration 대비 best score)
- 알고리즘 비교 모드 — 동일 입력으로 4개 동시 실행 후 스코어 표 출력
- 장애물 패턴에 "도로망(line buffer)" 추가
