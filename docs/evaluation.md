# 평가 가이드

## 서로 다른 세 평가

| 평가 | 데이터·대상 | 확인하는 것 |
| --- | --- | --- |
| 외부 벤치마크 | HPatches 580쌍 + UDIS-D 1,105쌍 × 5방법 | 동결 Q1 경로의 매칭·전역 H·APAP |
| 후처리 전 진단 | 프로젝트 Q1~E2, baseline 포함 25회 | 워핑된 원본 입력 사이의 중첩 PSNR·SSIM |
| 최종 PNG 평가 | 개선 4방법 × 5미션 = 저장 PNG 20개 | 최종 결과를 원본 좌표로 복원한 원본 일치도 |

외부 벤치마크는 Q3의 8장 BA나 E2 최종 합성 전체의 성능이 아닙니다. 최종 PNG의 PSNR·SSIM은 원본 일치도이며, 정답 기하·미관 점수가 아닙니다. 정답이 없는 프로젝트 사진과 UDIS-D의 Corner·독립점 오차는 산출하지 않았습니다.

## 결과 기록

점수와 해석은 [통합 보고서](PROJECT_REPORT.md)에 남겨 두었습니다. 아래 파일은 로컬 산출물로, Git에 포함되지 않습니다.
- 외부 벤치마크 요약 CSV (`results/evaluation/benchmark_summary.csv`)
- 영상 쌍별 상세 CSV (`results/evaluation/benchmark_pairs.csv`)
- 최종 PNG 20개 점수 CSV (`results/project_evaluation/summary.csv`)
- 후처리 전 진단 CSV (`results/evaluation/mission_summary.csv`)

외부 벤치마크 — 로컬 파일: `results/evaluation/benchmark_scores.png`

프로젝트 최종 PNG — 로컬 파일: `results/project_evaluation/project_scores.png`

HPatches에서는 MAGSAC 계열이 전역 H 정확도에 유리했습니다. APAP는 UDIS-D와 자체 Q2/Q3/E1/E2의 원본 일치도를 높였지만 HPatches의 독립점 오차는 B3보다 커졌습니다. 모든 지표에 대한 단일 우승 방법이라고 표현하지 않습니다.

## 참조 코드와 데이터

공식 데이터 안내와 UDIS++ 평가 참조 코드는 [third_party 서브모듈](../third_party/README.md)로 버전을 고정합니다. 외부 저장소의 모델을 실행하는 것이 아니라 본 프로젝트의 OpenCV·NumPy 평가기를 사용합니다. 서브모듈 초기화로 실제 이미지가 준비되지는 않으므로 [데이터 준비](../data/README.md)를 별도로 완료하세요.

## 재실행

외부 데이터 준비는 [데이터 안내](../data/README.md)를 따릅니다. 아래 명령은 기존 점수·보고서 파일을 다시 생성합니다. 평가 조건을 바꿨다면 별도의 결과 기록으로 구분하세요.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/evaluate_benchmarks.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/evaluate_missions.py
python -B experiments/report_evaluation.py
```

실제 저장 PNG 20개를 평가하려면 [비교 실험 실행](running.md)에 따라 baseline과 개선 네 방법을 모두 생성하세요. 새 clone에서는 `run_brown_lowe.py`와 `run_methods.py --method all --q all`로 이미지와 `experiment.json`을 함께 생성해야 합니다.

```bash
python -B experiments/evaluate_project_results.py
python -B experiments/plot_project_scores.py
```

최종 PNG 평가기는 입력·코드·PNG 해시와 재실행 픽셀 동일성을 검사합니다. 기존 측정 환경의 BLAS 설정을 유지해야 하며, 이 명령에는 일괄적으로 `OPENBLAS_NUM_THREADS=1`을 덧붙이지 않습니다. 차이가 나면 원본 PNG를 정답인 것처럼 교체하거나 검사를 무시하지 말고 환경과 기록을 확인합니다.

## 지표 읽기

- 독립점 Median·RMSE·P95: 작을수록 정확. 성공 사례 기준 통계이므로 실패 수도 같이 봅니다.
- Corner error: 전역 H의 네 모서리 평균 위치 오차. APAP 이전 H를 측정합니다.
- AUC@t: `100 × mean(max(1 - corner_error/t, 0))`. 실패는 0점. 임계값 성공률과 다릅니다.
- PSNR·SSIM: 높을수록 비교 영상과 유사. 조명·노출·보간·블렌딩의 영향도 포함됩니다.
- E1/E2: 전처리 전 진단은 같을 수 있으나, 최종 PNG 비교는 후처리 영향을 포함합니다.
- 시간: 현재 기록은 1회 실행 시간이며 반복 속도 벤치마크가 아닙니다.

프로토콜의 해시는 **당시 실행한 코드 버전**의 기록입니다. 이후 문서·실행기 정리로 파일 해시가 바뀌어도 과거 프로토콜을 새 코드의 해시로 덮어써서는 안 됩니다. 새 실행 시 새 기록을 생성합니다.
