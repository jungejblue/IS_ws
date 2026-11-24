# 정합·평가 산출물 관리

`results/` 전체를 Git에서 제외합니다. 기존 로컬 파일은 유지하지만 새 clone에는 포함되지 않습니다.

| 로컬 경로 | 내용 |
| --- | --- |
| `results/<방법>/` | Q1~E2 PNG, 미리보기, 실행 스냅샷·설정·해시, 상태·로그 |
| `results/evaluation/` | 외부 벤치마크와 미션 진단 점수·보고서·그래프 |
| `results/project_evaluation/` | 최종 PNG 평가·오차 지도·마스크 |
| `results/local/` | 개인 실행 결과 |

재현에 필요한 다섯 방법의 실행 소스는 [experiments/snapshots/](../experiments/snapshots/README.md)에 따로 보관합니다. [실행 안내](running.md)에 따라 정합 결과를 생성하고 [평가 안내](evaluation.md)에 따라 점수·보고서를 생성합니다. 과거 실험의 수치와 해석은 [통합 보고서](PROJECT_REPORT.md)에서 확인할 수 있습니다.

이미지·평가 파일은 별도 파일 공유 경로로 전달하고, PR 본문에는 실행 조건과 요약 점수를 적습니다. `git add -f`로 산출물을 강제로 추가하지 않습니다.
