# 실행과 실험 재현

설치는 [README](../README.md), 사진과 벤치마크 준비는 [데이터 안내](../data/README.md)를 따릅니다. 모든 명령은 저장소 루트 기준입니다.

## 한 방법·한 미션 실행

```bash
python -B run.py --q 1 \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --input /path/to/two-photos \
  --out results/local/apap
```

`--input`을 생략하면 Q1은 `data/Q1`, Q2는 `data/Q2`, Q3·E1·E2는 `data/Q3`를 사용합니다. `--q`에는 `1`, `2`, `3`, `e1`, `e2`, `all`을 지정합니다. 전체 미션을 실행할 때는 `--input`을 생략하고 세 폴더를 모두 준비하세요.

`run.py`는 PNG와 상태 JSON만 저장합니다. 최종 PNG 평가에 필요한 코드·입력 해시와 실행 메타데이터는 아래 방법별 실행기로 생성합니다.

## 고정 코드로 비교 실험 재현

```bash
# 다섯 방법 × 다섯 미션 = PNG 25개
python -B experiments/run_methods.py --method all --q all

# 기준 방법 또는 특정 개선 방법만 실행
python -B experiments/run_methods.py --method brown_lowe --q all
python -B experiments/run_methods.py --method rootsift_magsac_apap --q all

# 입력·환경을 빠르게 확인하는 Q1 실행
python -B experiments/run_methods.py --method all --q 1
```

| `--method` | 고정 소스 (`experiments/snapshots/`) | 결과 폴더 (`results/`) |
| --- | --- | --- |
| `brown_lowe` | `brown+lowe.py` | `brown+lowe/` |
| `rootsift` | `RootSIFT.py` | `RootSIFT/` |
| `magsac` | `MAGSAC++.py` | `MAGSAC++/` |
| `rootsift_magsac` | `RootSIFT+MAGSAC++.py` | `RootSIFT+MAGSAC++/` |
| `rootsift_magsac_apap` | `RootSIFT+MAGSAC+++APAP.py` | `RootSIFT+MAGSAC+++APAP/` |

기본 `--source frozen`은 고정 소스를 바이트 그대로 복사해 실행합니다. 실행 기록에 실제 소스 경로·해시·입력 순서·해시·환경을 저장합니다. 동일한 절차를 자신의 사진으로 수행할 수 있지만, 사진이나 환경이 다르면 과거 이미지·점수와 일치한다고 보장하지 않습니다.

## 수정한 코드로 새 실험

루트 `stitching.py`를 수정한 뒤 다음과 같이 실행합니다.

```bash
python -B experiments/run_methods.py --source current --method rootsift_magsac_apap --q all
```

이 모드는 현재 루트 소스에 비교 프로필과 선택한 방법 설정을 추가해 실행합니다. 고정 버전 재현과 구분하도록 `source_mode`가 기록됩니다. 기준 방법도 `--method brown_lowe`로 명시적으로 선택합니다.

## 출력과 보존

각 방법 폴더에는 미션별 `Q1_result.png` 등의 이미지, `Q1_status.json`, `Q1_run.log`, `previews/`, 실행 소스 `stitching_snapshot.py`, 메타데이터 `experiment.json`이 생성됩니다. `results/comparison.html`은 방법별 결과 갤러리입니다.

실행기는 선택한 방법 폴더를 갱신합니다. `--q 1` 같은 부분 실행도 해당 방법의 `experiment.json`을 새로 쓰므로, 전체 평가 전에는 `--q all`로 생성하세요. 보존할 실험은 다른 경로에 복사한 후 실행합니다. `results/` 전체는 Git에서 제외되며 새 clone에는 생성 결과가 없습니다.

`[OK]`는 함수 실행과 반환 형식이 통과했다는 의미입니다. 정합 성능은 [평가 절차](evaluation.md)와 영상 관찰로 확인합니다.

## 보조 도구와 오류 확인

| 명령 | 용도 |
| --- | --- |
| `python -B scripts/check_environment.py` | Python·OpenCV 필수 API 확인 |
| `python -B scripts/check_environment.py --data` | Q1·Q2·Q3 사진 장수와 디코딩 확인 |
| `python -B scripts/check_repository.py` | 제출 인터페이스·문서 링크·Git 추적 정책 검사 |
| `python -B scripts/prepare_benchmarks.py` | 배포본 해시·경로 검사 후 압축 해제 |

사진은 `.jpg`, `.jpeg`, `.png`, `.bmp`를 지원합니다. 환경 검사와 정합 실행에는 OpenCV 배포판을 하나만 설치하세요. `opencv-python`과 `opencv-contrib-python-headless`를 중복 설치하면 동일한 `cv2` 모듈을 공유합니다. 이 프로젝트는 `requirements.txt`의 headless contrib 패키지를 사용합니다.

Q3 실패 시 장수·순서·중첩 영역을 먼저 확인하세요. 같은 결과를 재현할 때는 라이브러리 버전과 BLAS 스레드 설정도 유지해야 합니다. Matplotlib 캐시 경로가 쓰기 불가능하면 `MPLCONFIGDIR=/tmp/is-ws-matplotlib`처럼 쓰기 가능한 경로를 지정합니다.
