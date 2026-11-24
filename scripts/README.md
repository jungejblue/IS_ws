# 개발·데이터 도구

| 명령 | 역할 |
| --- | --- |
| `python scripts/check_environment.py` | Python·의존성·OpenCV 필수 API 검사 |
| `python scripts/check_environment.py --data` | Q1/Q2/Q3 장수·파일명·디코딩 검사 |
| `python scripts/check_environment.py --data --reference-inputs` | 게시된 결과 원본과 해시 비교 |
| `python scripts/check_repository.py` | 과제 인터페이스·로컬 링크·Git 추적 정책 검사 |
| `python scripts/prepare_benchmarks.py` | 다운로드 압축 해시·안전 경로 검증 후 해제 |

여기 있는 스크립트는 개발 보조 도구이며 제출 파일에서 import하지 않습니다. 대용량 다운로드는 [experiments/download_benchmarks.py](../experiments/download_benchmarks.py), 입력 경로는 [data/README.md](../data/README.md)를 참고하세요.
