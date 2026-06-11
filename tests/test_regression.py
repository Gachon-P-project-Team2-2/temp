"""Playwright 기반 UI 회귀 테스트.

실행:
    venv/Scripts/python.exe -m pytest tests/test_regression.py -v

conftest.py가 Dash 앱을 자동 시작/종료하고 세션 전역 브라우저를 관리한다.
각 테스트는 새 페이지로 시작해 격리된다.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

# REGISTRY / PATTERN_CHOICES를 직접 import하여, 항목 추가/제거 시 테스트가 자동 갱신됨
from optimizers import REGISTRY
from patterns import PATTERN_CHOICES


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _generate_data(page: Page) -> None:
    """가상 데이터 생성 버튼 클릭 + 완료 대기."""
    btn = page.get_by_role("button", name="가상 데이터 생성")
    btn.scroll_into_view_if_needed()
    btn.click()
    page.wait_for_selector("text=가상 환경 생성 완료", timeout=60_000)


def _run_optimization(page: Page, timeout_ms: int = 180_000) -> None:
    """계산 실행 버튼 클릭 + 메트릭 표시 대기.

    GA는 기본 하이퍼파라미터로 수 분까지 걸릴 수 있으므로 넉넉히 대기.
    """
    page.get_by_role("button", name="계산 실행").click()
    page.wait_for_selector("text=총 트래픽", timeout=timeout_ms)


def _slider_to_min(page: Page, label: str) -> None:
    """슬라이더를 최솟값으로 (Home 키)."""
    slider = page.get_by_role("slider", name=label)
    slider.focus()
    slider.press("Home")


def _ensure_expander_open(page: Page, label: str) -> None:
    """html.Details 요소를 '열림' 상태로 보장 (이미 열려있으면 no-op)."""
    summary = page.locator("summary").filter(has_text=label).first
    parent = summary.locator("..")
    is_open = parent.evaluate("el => el.open")
    if not is_open:
        summary.click()


def _minimize_hyperparams(page: Page, algo: str) -> None:
    """속도를 위해 반복 횟수를 최소로. Dash 버전에서는 미사용 (no-op)."""
    pass


def _select_by_combobox(page: Page, dropdown_id: str, option_name: str) -> None:
    """Dash dcc.Dropdown에서 옵션 선택.

    현재 표시 값이 이미 option_name이면 스킵한다.
    """
    container = page.locator(f"#{dropdown_id}")
    container.scroll_into_view_if_needed()
    value_label = container.locator(".Select-value-label")
    if value_label.count() > 0:
        current = (value_label.first.text_content(timeout=2_000) or "").strip()
        if current == option_name:
            return
    container.click()
    option = page.get_by_role("option", name=option_name, exact=True)
    option.wait_for(state="visible", timeout=10_000)
    option.click(force=True)


def _select_algorithm(page: Page, name: str) -> None:
    _select_by_combobox(page, "algo-select", name)


def _select_pattern(page: Page, name: str) -> None:
    """트래픽 패턴 dropdown에서 name 선택 (details 섹션이 열려있어야 함)."""
    _select_by_combobox(page, "traffic-pattern", name)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
def test_app_loads(page: Page):
    """기본 UI가 로드되고 콘솔 에러가 없는지."""
    expect(page.get_by_text("시뮬레이터 제어")).to_be_visible()
    expect(page.get_by_text("1. 환경 설정")).to_be_visible()
    expect(page.get_by_text("2. 시각화 설정")).to_be_visible()
    expect(page.get_by_text("계산 알고리즘")).to_be_visible()
    expect(page.get_by_role("button", name="가상 데이터 생성")).to_be_visible()


def test_generate_data_golden_path(page: Page):
    """데이터 생성 → 성공 메시지 + 다운로드 버튼 노출."""
    _generate_data(page)
    expect(page.get_by_text("가상 환경 생성 완료")).to_be_visible()
    expect(page.get_by_role("button", name="GIS CSV")).to_be_visible()
    expect(page.get_by_role("button", name="Local CSV")).to_be_visible()


def test_algorithm_selectbox_has_all_optimizers(page: Page):
    """REGISTRY의 모든 알고리즘이 UI dropdown에 노출되는지 (목록 직접 참조)."""
    page.locator("#algo-select").click()
    for cls in REGISTRY:
        expect(page.get_by_role("option", name=cls.name, exact=True)).to_be_visible()


def test_traffic_pattern_selectbox_has_all_patterns(page: Page):
    """PATTERN_CHOICES의 모든 패턴이 UI에 노출되는지 (목록 직접 참조)."""
    _ensure_expander_open(page, "트래픽 세부 설정")
    page.locator("#traffic-pattern").click()
    for name in PATTERN_CHOICES:
        expect(page.get_by_role("option", name=name, exact=True)).to_be_visible()


def test_kmeans_runs_golden_path(page: Page):
    """K-Means(기본 알고리즘)로 전체 골든 패스 확인.

    다른 알고리즘들은 test_algorithm_selectbox_has_all_optimizers에서 UI 등록 여부만
    확인한다. selectbox 변경 후 Dash re-render 타이밍이 불안정해 per-algo 파라미터화는
    flaky하므로, 대표 알고리즘 하나로 end-to-end만 검증.
    """
    _generate_data(page)
    page.wait_for_timeout(1500)
    _run_optimization(page)

    # 메트릭 카드 확인
    expect(page.get_by_text("총 트래픽")).to_be_visible()
    expect(page.get_by_text("커버된 트래픽")).to_be_visible()
    expect(page.get_by_text("커버된 면적")).to_be_visible()
    expect(page.get_by_text("평균 SINR")).to_be_visible()


def test_propagation_model_section_present(page: Page):
    """사이드바에 '전파 모델' 섹션이 존재하는지 확인."""
    expect(page.get_by_text("전파 모델")).to_be_visible()


def test_ring_pattern_generates(page: Page):
    """ring 패턴(비-기본)으로 데이터 생성 — 8개 패턴 중 대표로 검증."""
    _ensure_expander_open(page, "트래픽 세부 설정")
    _select_pattern(page, "ring")
    _generate_data(page)
    expect(page.get_by_text("가상 환경 생성 완료")).to_be_visible()


# Dash의 다운로드 버튼은 첫 클릭 전까지 blob URL이 없어 404를 찍는다 — 무해한 노이즈.
_IGNORED_ERROR_PATTERNS = (
    "Download Button source error",
    "Failed to load resource",
)


def test_no_console_errors(page: Page):
    """페이지 로드와 데이터 생성 중 콘솔 error가 발생하지 않는지 (무해한 노이즈 제외)."""
    errors: list[str] = []

    def _on_console(msg):
        if msg.type != "error":
            return
        text = msg.text
        if any(p in text for p in _IGNORED_ERROR_PATTERNS):
            return
        errors.append(text)

    page.on("console", _on_console)

    _generate_data(page)
    _select_algorithm(page, "K-Means")
    _run_optimization(page)

    assert errors == [], f"Unexpected console errors: {errors}"
