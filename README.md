# IS_ws — Image Stitching

OpenCV와 NumPy로 평면·원통·구면 파노라마를 만들고, 특징 기술자와 변환 추정 방법을 비교하는 프로젝트입니다. Brown–Lowe 기반 기준 방법에서 시작해 RootSIFT, MAGSAC++, APAP를 차례로 적용합니다. `cv2.Stitcher` 대신 특징점 검출부터 합성까지 각 단계를 직접 구성합니다.

## 빠른 시작

Python 3.10 이상을 사용합니다. GPU는 필요하지 않습니다. 모든 명령은 저장소 루트에서 실행합니다.

```bash
git clone --recurse-submodules https://github.com/jungejblue/IS_ws.git
cd IS_ws
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B scripts/check_environment.py
```

저장소에 포함된 `data/Q1/` 사진 두 장으로 바로 실행할 수 있습니다. 사진은 파일명의 숫자 순서로 읽습니다.

```bash
python -B run.py --q 1 \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --out results/local/apap
```

결과는 `results/local/apap/Q1_result.png`, 실행 상태는 `Q1_status.json`에 저장됩니다. 다른 폴더의 사진은 `--input /path/to/photos`로 지정합니다.

## 미션과 입력

| 미션 | 사진 수 | 출력 |
| --- | ---: | --- |
| Q1 | 2 | 평면 정합 |
| Q2 | 3 | 평면 합성 후 전체 파노라마를 원통 좌표로 재투영 |
| Q3 | 8 | 원통 파노라마 |
| E1 | Q3와 동일 | 구면 파노라마 |
| E2 | Q3와 동일 | 구면 파노라마에 노출 보정과 블렌딩 적용 |

Q3는 2행 4열의 촬영 순서를 가정합니다. 넓은 시야로 하나의 평면에 담기 어려운 Q3·E1·E2는 공통 원통·구면 좌표에서 합성합니다. 입력 조건은 [데이터 안내](data/README.md)를 참고하세요.

## 실험 재현

1. 저장소에 포함된 `data/Q1`, `data/Q2`, `data/Q3`의 입력 사진을 사용합니다.
2. 아래 명령으로 기준 방법과 개선 네 방법을 실행합니다. 다섯 미션씩 총 25개의 PNG와 실행 기록을 생성합니다.
3. [평가 안내](docs/evaluation.md)에 따라 외부 벤치마크 또는 최종 PNG 평가를 실행합니다.

```bash
python -B scripts/check_environment.py --data
python -B experiments/run_methods.py --method all --q all
```

실행기는 기본적으로 `experiments/snapshots/`의 고정 코드를 사용합니다. 루트 `stitching.py`를 수정해 새 실험을 하려면 `--source current`를 붙입니다. 두 실행 방식의 차이와 방법별 선택은 [실행 안내](docs/running.md)에 설명되어 있습니다.

프로젝트 입력 사진 13장은 저장소에 포함됩니다. 외부 벤치마크 데이터와 생성 결과는 포함되지 않습니다. 자신의 사진으로도 같은 절차를 실행할 수 있으며, 기존 결과와 픽셀·점수를 비교할 때는 입력뿐 아니라 실행 환경도 동일해야 합니다.

## 문서

| 문서 | 내용 |
| --- | --- |
| [알고리즘](docs/algorithm.md) | 처리 흐름, 방법별 차이, 투영·블렌딩, 구현 한계 |
| [실행](docs/running.md) | 고정 코드 재현, 새 코드 실험, 출력·오류 확인 |
| [평가](docs/evaluation.md) | 벤치마크 준비 이후 실행 순서와 지표 해석 |
| [데이터](data/README.md) | 사진 순서, 외부 데이터 다운로드·검증 |
| [서브모듈](third_party/README.md) | 공식 참조 저장소 초기화·버전 관리 |
| [과제 조건](docs/assignment.md) | 이 구현의 기반이 된 과제 요구사항과 원문 |

## 디렉터리

```text
IS_ws/
├── stitching.py          # 개발·제출용 단일 파일 알고리즘
├── run.py                # 제공 실행기
├── requirements.txt      # Python 의존성
├── experiments/          # 방법별 실행·평가·보고서 생성
│   └── snapshots/        # 다섯 방법의 고정 실행 소스
├── scripts/              # 환경·저장소 검사, 데이터 압축 해제
├── docs/                 # 알고리즘·실행·평가·과제 안내
├── data/                 # 입력 사진 포함, 외부 벤치마크는 Git 제외
├── third_party/          # 공식 참조 저장소의 서브모듈
└── results/              # 생성 결과, 전체 Git 제외
```

`python -B scripts/check_repository.py`로 제출 인터페이스·문서 링크·Git 제외 정책을 확인할 수 있습니다. `results/`와 데이터 ZIP을 강제로 Git에 추가하지 마세요.
