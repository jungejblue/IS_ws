# 실험 실행기

알고리즘은 루트 `stitching.py`에 있고 이 폴더는 실험·평가용입니다. 아래 스크립트들은 제출하지 않습니다. 명령은 저장소 루트에서 실행합니다.

| 파일 | 역할 | 주요 인자·출력 |
| --- | --- | --- |
| `run_brown_lowe.py` | 현재 코드로 baseline 생성 | `results/brown+lowe/` 갱신 |
| `run_methods.py` | 현재 코드로 개선 방법 생성 | `--method all/rootsift/magsac/rootsift_magsac/rootsift_magsac_apap`, `--q all/1/2/3/e1/e2` |
| `download_benchmarks.py` | 공개 데이터 압축·출처 다운로드 | `hpatches`, `drive-file --id ... --name ...` |
| `evaluate_benchmarks.py` | 동결 Q1 경로로 외부 평가 | `--workers 4`, `--dataset all/hpatches/udis_d`, `--limit N` |
| `evaluate_missions.py` | 25개 미션의 후처리 전 진단 | `results/evaluation/mission_photometric.json` |
| `report_evaluation.py` | 외부 평가·진단 보고서 | CSV·HTML·PNG·PDF |
| `evaluate_project_results.py` | 저장 PNG 20개 원본 일치도 | `--pilot`은 APAP의 5개 미션 좌표 재현 검사 |
| `plot_project_scores.py` | 최종 PNG 점수 시각화 | PNG·PDF·보고서 그래프 삽입 |

테스트 코드와 테스트용 CI는 삭제했습니다. 생성되는 `results/` 전체는 Git에서 제외합니다. 재현용 실행 소스는 [snapshots/](snapshots/README.md)에 보관합니다. 외부 벤치마크·후처리 전 진단은 로컬 실행 스냅샷이 없으면 이 소스를 사용합니다.

생성·평가 스크립트는 지정된 기존 결과 경로를 갱신합니다. `--limit`으로 작은 평가를 실행해도 기존 요약 파일을 덮어쓸 수 있으므로 공식 기록을 보존해야 할 때는 별도 작업 사본에서 실행하세요. `--resume`은 동일 입력·설정·코드에서만 사용하며 자동으로 실험 호환성을 보장하지 않습니다.

과거 결과만 재현할 때는 실행기를 통해 스냅샷을 다시 만들지 말고 [동결 코드 실행법](../docs/running.md)을 사용합니다. [평가 프로토콜](../docs/evaluation.md)
