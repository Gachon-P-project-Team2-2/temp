"""메타휴리스틱 알고리즘 모음.

각 알고리즘은 Optimizer ABC를 구현하며, 내부적으로 _shared 모듈의
calculate_score / perturb / clip 등의 유틸리티를 공유한다.

_shared는 이 패키지 내부 전용 (private) — 외부에서 import 금지.
"""
