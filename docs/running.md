# 실행과 실험 재현

모든 명령은 저장소 루트에서 실행합니다. [입력 사진 준비](../data/README.md)를 먼저 완료하세요.

## 루트 코드와 동결 스냅샷

`stitching.py`는 개발·제출 대상이고 `run.py`는 수정하지 않는 과제 실행기입니다. 실험별 `experiments/snapshots/<방법>.py`는 당시 설정까지 포함한 재현용 파일입니다.

```bash
# 현재 개발 코드: 루트 기본 프로필 사용
python -B run.py --q 1 --out results/local/dev

# 발표 비교에 사용한 APAP 프로필
python -B run.py --q all \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --out results/local/apap

# 한 미션을 다른 입력으로 실행
python -B run.py --q 3 --input /path/to/eight-photos \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --out results/local/new-q3
```

`--q`는 `1`, `2`, `3`, `e1`, `e2`, `all`입니다. 실행기는 번호를 자연순으로 정렬하며 `[OK]`는 함수 실행·반환 형식 통과를 의미합니다.

| 방법 | 코드 경로 |
| --- | --- |
| Brown–Lowe 기반 baseline | `experiments/snapshots/brown+lowe.py` |
| RootSIFT | `experiments/snapshots/RootSIFT.py` |
| MAGSAC++ | `experiments/snapshots/MAGSAC++.py` |
| RootSIFT + MAGSAC++ | `experiments/snapshots/RootSIFT+MAGSAC++.py` |
| 위 조합 + APAP | `experiments/snapshots/RootSIFT+MAGSAC+++APAP.py` |

새 실험의 출력에는 `results/local/<이니셜>/<실험명>`을 사용합니다. 비교용 기존 스냅샷을 직접 고치지 말고 루트 코드를 수정하거나 새 스냅샷을 만드세요.

## 기존 고해상도 PNG 복원

Git에는 재현용 실행 소스와 입력 manifest만 있습니다. 이미지·실행 메타데이터·평가 기록은 포함되지 않습니다. 아래 명령은 동결 코드로 PNG를 다시 생성합니다. 기존 메타데이터가 로컬에 남아 있다면 같은 입력·환경에서 비교할 수 있습니다.

```bash
python -B run.py --q all --code experiments/snapshots/RootSIFT.py --out results/RootSIFT
python -B run.py --q all --code experiments/snapshots/MAGSAC++.py --out results/MAGSAC++
python -B run.py --q all --code experiments/snapshots/RootSIFT+MAGSAC++.py --out results/RootSIFT+MAGSAC++
python -B run.py --q all --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py --out results/RootSIFT+MAGSAC+++APAP
python -B run.py --q all --code experiments/snapshots/brown+lowe.py --out results/brown+lowe
```

사진이나 라이브러리가 다르면 파일이 생성되더라도 기존 `experiment.json`의 해시와 일치하지 않을 수 있습니다. 그런 결과를 기존 정량 평가와 같은 실행이라고 기록하지 않습니다.

## 비교 실험을 새로 생성할 때

```bash
python -B experiments/run_methods.py --method rootsift_magsac_apap --q all
python -B experiments/run_methods.py --method all --q all
python -B experiments/run_brown_lowe.py
```

이 실행기들은 **현재 루트 `stitching.py`에서 스냅샷·메타데이터·결과를 다시 생성하고 해당 방법 폴더를 갱신**합니다. 과거 실험 재생과 다릅니다. 갱신이 목적일 때만 실행하고, PR에서 코드·입력·설정·점수를 함께 검토합니다. `run_methods.py`의 일부 미션 실행은 해당 실행의 manifest만 다시 기록하므로, 비교 기록을 공식 갱신할 때는 `--q all`을 사용합니다.

## 출력

```text
results/local/apap/
├─ Q1_result.png ... E2_result.png
└─ Q1_status.json ... E2_status.json
```

`run.py`는 `experiment.json`을 만들지 않습니다. 최종 PNG 평가에 필요한 전체 실험 메타데이터와 미리보기 생성은 `experiments/` 실행기가 담당합니다. 새 clone에서 평가하려면 입력을 준비하고 `experiments/run_brown_lowe.py`와 `experiments/run_methods.py --method all --q all`을 모두 실행하세요. [평가 재현](evaluation.md)
