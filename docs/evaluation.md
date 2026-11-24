# 평가 가이드

## 평가 범위

| 평가 | 입력 | 측정 대상 |
| --- | --- | --- |
| 외부 벤치마크 | HPatches·UDIS-D 영상 쌍 | 다섯 방법의 Q1 매칭·전역 호모그래피·APAP |
| 후처리 전 진단 | 사용자가 준비한 Q1/Q2/Q3 사진 | 워핑된 입력 간 중첩 PSNR·SSIM |
| 최종 PNG 평가 | 개선 네 방법으로 생성한 Q1·Q2·Q3·E1·E2 이미지 | 최종 결과를 원본 좌표로 복원한 원본 일치도 |

외부 벤치마크는 Q3의 8장 카메라 최적화나 E2의 최종 합성 전체를 평가하지 않습니다. 최종 PNG의 PSNR·SSIM은 입력과의 일치도이며 정답 기하나 미관 점수가 아닙니다. 정답 호모그래피가 없는 UDIS-D와 사용자 입력에는 정답 기준 Corner·독립점 오차를 산출하지 않습니다.

## 데이터와 참조 코드

[서브모듈](../third_party/README.md)은 공식 데이터 안내와 평가 참조 코드를 제공합니다. 실제 데이터는 [데이터 준비 안내](../data/README.md)에 따라 별도로 다운로드합니다. 평가에는 본 프로젝트의 OpenCV·NumPy 코드를 사용하며 외부 딥러닝 모델을 실행하지 않습니다.

## 외부 벤치마크와 입력 진단

전체 평가는 두 벤치마크 데이터를 모두 준비한 후 저장소 루트에서 실행합니다. `--dataset hpatches` 또는 `--dataset udis_d`로 한 데이터셋만 선택할 수도 있습니다. 입력 진단에는 `data/Q1`, `data/Q2`, `data/Q3`도 필요합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/evaluate_benchmarks.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/evaluate_missions.py
python -B experiments/report_evaluation.py
```

평가기는 방법별 로컬 실행 스냅샷이 있으면 해당 코드를, 없으면 `experiments/snapshots/`의 코드를 사용합니다. 고정 코드 비교를 원하면 로컬 실행 기록도 `--source frozen`으로 생성하세요. 선택된 코드·입력·조건은 평가 기록에서 확인할 수 있습니다.

## 사용자 입력으로 최종 PNG 평가

[입력 준비](../data/README.md)를 완료한 후 이미지와 실행 메타데이터를 함께 생성합니다. 아래 실행기는 기존 방법별 결과를 갱신하므로 보존할 실험은 별도 경로에 복사하세요.

```bash
python -B experiments/run_methods.py --method all --q all
python -B experiments/evaluate_project_results.py
python -B experiments/plot_project_scores.py
```

생성은 baseline 포함 25개 이미지이며, 최종 PNG 평가 대상은 개선 네 방법 × 다섯 미션의 20개입니다. 평가기는 입력·코드·PNG 해시와 재실행 이미지의 픽셀 동일성을 검사합니다. 생성과 평가에서 동일한 라이브러리·BLAS 스레드 설정을 사용하세요. 불일치하면 환경과 입력을 확인하고 검증을 생략하지 않습니다.

## 실행 옵션과 결과 보존

`--workers`는 병렬 프로세스 수, `--limit N`은 선택한 데이터셋에서 평가할 최대 영상 쌍 수입니다. 제한 실행의 점수는 전체 데이터셋 점수가 아닙니다. `--resume`은 기존 입력 선택·코드·환경·평가 조건과 일치하는 경우에만 이어서 실행하며, 작업자 수만 변경할 수 있습니다.

모든 평가 도구는 고정된 `results/evaluation/` 또는 `results/project_evaluation/`에 기록합니다. 작은 확인용 실행도 기존 결과를 갱신하므로 보존할 결과는 먼저 복사하세요. 보고서 생성은 해당 평가가 완료된 뒤 실행합니다.

## 출력

| 경로 | 내용 |
| --- | --- |
| `results/evaluation/benchmark_summary.csv` | 외부 벤치마크 방법별 요약 |
| `results/evaluation/benchmark_pairs.csv` | 영상 쌍별 상세 점수 |
| `results/evaluation/mission_summary.csv` | 입력 영상의 후처리 전 진단 |
| `results/evaluation/report.html` | 외부 평가·진단 보고서 |
| `results/project_evaluation/summary.csv` | 최종 PNG 점수 |
| `results/project_evaluation/report.html` | 최종 PNG 평가 보고서·오차 지도 |

생성된 점수·보고서·이미지는 모두 Git에서 제외합니다. 저장소에는 작성자의 평가 결과가 포함되어 있지 않습니다.

## 지표 해석

- 독립점 Median·RMSE·P95: 작을수록 정답 좌표에 가깝습니다. 성공 사례의 오차와 실패 수를 함께 확인합니다.
- Corner error: 전역 호모그래피로 변환한 네 모서리의 평균 위치 오차입니다. APAP 이전의 전역 변환을 평가합니다.
- AUC@t: `100 × mean(max(1 - corner_error/t, 0))`. 실패는 0점이며 임계값 성공률과 다릅니다.
- PSNR·SSIM: 높을수록 참조 영상과 유사합니다. 조명·노출·보간·블렌딩에도 영향을 받습니다.
- 시간: 한 번의 실행 시간은 반복 측정한 속도 벤치마크와 구분합니다.

방법을 비교할 때는 같은 입력·평가 영역·후처리·실패 처리 조건을 유지하세요. 입력이나 코드를 바꾸면 이전 점수를 덮어 해석하지 말고 새로운 평가 기록을 생성합니다.

## 도구 구성

| 파일 (`experiments/`) | 역할 |
| --- | --- |
| `run_methods.py` | 기준 방법과 개선 네 방법의 결과·메타데이터 생성 |
| `evaluate_benchmarks.py` | HPatches·UDIS-D 영상 쌍 평가 |
| `evaluate_missions.py` | 다섯 미션의 후처리 전 중첩 영역 진단 |
| `report_evaluation.py` | 외부 평가·미션 진단의 CSV·HTML·그래프 생성 |
| `evaluate_project_results.py` | 최종 PNG 20개 평가·오차 지도 생성; `--pilot`은 APAP 방법의 재현 확인 |
| `plot_project_scores.py` | 최종 PNG 점수 그래프 생성 |
