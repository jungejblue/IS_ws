# IS_ws — Image Stitching

OpenCV·NumPy로 **Q1~E2 이미지 정합 과제**를 구현하고, 특징 기술자·강건 추정·국소 워핑을 비교하는 팀 저장소입니다.

- **구현:** 평면·원통·구면 정합, RootSIFT, MAGSAC++, APAP 계열 Moving DLT, gain·multi-band 합성.
- **실험:** Brown–Lowe 기반 과제용 baseline + 개선 4개 방법, 미션별 총 25개 결과.
- **평가:** HPatches·UDIS-D 외부 영상 쌍 8,425회 평가와 프로젝트 최종 PNG 20개 원본 일치도 평가.

현재 제출 알고리즘은 한 파일로 유지합니다. 루트 `stitching.py`의 기본 설정과 저장된 실험 스냅샷 설정은 다릅니다. 아래 빠른 실행은 **저장된 APAP 방법**을 사용합니다. [방법·평가 차이](docs/evaluation.md)

## 문서 안내

| 목적 | 문서 |
| --- | --- |
| 처음 설치하고 실행하기 | [설치·환경 확인](docs/setup.md), [실행·실험 재현](docs/running.md) |
| 코드·스크립트 찾기 | [실험 실행기](experiments/README.md), [개발 도구](scripts/README.md) |
| 입력 준비와 다운로드 | [데이터 안내](data/README.md) |
| 결과·지표 이해하기 | [평가 가이드](docs/evaluation.md), [산출물 관리](docs/results.md) |
| 팀 협업하기 | [Git·PR 사용법](docs/git_guide.md), [개발 주의사항](docs/notes.md) |
| 과제·제출 확인 | [공지 요약과 제출 조건](docs/assignment.md), [공식 PDF](docs/assignment/이미지스티칭_프로젝트_안내.pdf) |
| 전체 과정과 상세 수치 | [프로젝트 통합 보고서](docs/PROJECT_REPORT.md) |

## 1. Clone·설치

명령은 저장소 루트에서 실행합니다. Python 3.10 이상이 필요하며 GPU는 사용하지 않습니다.

```bash
git clone --recurse-submodules https://github.com/jungejblue/IS_ws.git
cd IS_ws
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

원본 사진·외부 벤치마크 데이터·모든 정합 및 평가 산출물은 Git에서 제외합니다. 공식 벤치마크 안내·참조 코드는 `third_party/`의 서브모듈로 관리합니다. Q1/Q2는 과제 제공 사진, Q3는 **팀 내부에서 공유한 직접 촬영 8장**을 `data/Q1`, `data/Q2`, `data/Q3`에 넣으세요. 파일 순서와 해시 확인은 [데이터 안내](data/README.md)를 따릅니다.

```bash
python scripts/check_environment.py --data
```

패키지 버전에 따라 기존 결과와 미세한 수치 차이가 생길 수 있습니다. 기존 측정 환경은 Python 3.10.12 / OpenCV 4.5.4 / NumPy 1.26.4 / Matplotlib 3.5.1입니다. 현재 requirements는 설치 가능한 개발 범위이며 기존 환경의 완전한 lockfile은 아닙니다.

### 벤치마크 서브모듈과 데이터 준비

| 서브모듈 | 역할 |
| --- | --- |
| `third_party/hpatches-dataset` | HPatches 공식 데이터 안내 |
| `third_party/udis-d` | UDIS-D 공식 배포 안내·원 논문 코드 |
| `third_party/udis2` | 기존 평가에서 참고한 UDIS++ 평가 코드 |

이미 clone했다면 다음 명령으로 저장된 커밋의 서브모듈을 가져옵니다.

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

**서브모듈에는 전체 벤치마크 데이터가 포함되지 않습니다.** 외부 평가를 실행할 때만 아래 명령으로 다운로드하고 압축을 해제합니다. 이미 준비된 로컬 데이터는 그대로 사용합니다.

```bash
python experiments/download_benchmarks.py hpatches
python experiments/download_benchmarks.py drive-file \
  --id 1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0 --name udis_testing.zip
python scripts/prepare_benchmarks.py
```

이미지·ZIP은 `data/benchmarks/`, 점수·보고서는 `results/`에 저장하며 모두 Git에서 제외합니다. 서브모듈은 참고용으로, 과제 실행 시 외부 딥러닝 모델이나 해당 저장소의 의존성을 설치·실행하지 않습니다. [서브모듈 관리](third_party/README.md) · [데이터 검증](data/README.md) · [평가 실행](docs/evaluation.md)

## 2. Q1부터 실행

사진을 준비한 뒤, 발표 비교에 사용한 RootSIFT + MAGSAC++ + APAP 스냅샷으로 실행합니다.

```bash
python -B run.py --q 1 \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --out results/local/apap

# Q1, Q2, Q3, E1, E2 전체
python -B run.py --q all \
  --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py \
  --out results/local/apap
```

`results/local/apap/`에 `Q1_result.png`~`E2_result.png`와 실행 상태가 저장됩니다. Q3/E1/E2는 같은 8장을 사용합니다. Q1/Q2/Q3/E1은 후처리 없이 합성하고, E2만 global gain·max-weight mask·multi-band를 적용합니다.

`python run.py --q all`은 **개발 중인 루트 코드**를 실행합니다. 기본값은 `ALGORITHM="brown_lowe"`, `BASELINE_PROFILE=None`이므로 위 스냅샷과 같은 실험이라고 해석하지 않습니다. [설정별 실행 방법](docs/running.md)

## 3. 데이터 없이 검사

```bash
python scripts/check_repository.py
```

저장소 검사는 제출 인터페이스·문서 링크·Git 추적 정책을 확인합니다. 실제 사진의 성능은 별도 미션 실행으로 확인합니다. 테스트 소스와 테스트용 CI 워크플로는 삭제했습니다. 캐시·임시 파일과 모든 실행 산출물은 원격에 포함하지 않습니다.

## Repository 레이아웃

| 경로 | 역할 | Git 정책 |
| --- | --- | --- |
| `stitching.py` | 과제 제출용 단일 파일 알고리즘 | 버전 관리 |
| `run.py` | 조교 제공 실행기, 수정 금지 | 버전 관리 |
| `experiments/` | 비교 실험·다운로드·평가·보고서 도구 | 버전 관리 |
| `scripts/` | 환경·저장소 확인, 벤치마크 압축 해제 | 버전 관리 |
| `third_party/` | 공식 벤치마크 안내·참조 코드 | 서브모듈 커밋 관리 |
| `docs/` | 설치·실행·협업·과제·전체 보고서 | 버전 관리 |
| `data/` | 과제 사진과 외부 데이터셋 | 안내·입력 manifest·벤치마크 배포 설정만 관리 |
| `experiments/snapshots/` | 과거 다섯 방법의 실행 소스 | 버전 관리 |
| `results/` | 정합 이미지·미리보기·평가 점수·보고서·로그 | 전체 Git 제외 |
| `submission/`, `tmp/` | 제출 압축·임시 작업 | Git 제외 |

`container_localization`의 빠른 실행·검증 안내, `nrf_ws`의 문서 목차, `save_ai_ws_2026`의 역할별 디렉터리·Git 협업 구성을 참고했습니다. 과제의 단일 파일·고정 실행기 조건 때문에 `stitching.py`와 `run.py`는 루트에 둡니다.

## 협업 규칙

본인 이니셜과 작업명을 포함한 브랜치에서 수정하고 PR로 병합합니다. 예: `jj/docs-readme`, `jj/fix-matching`. 새 실험은 `results/local/<이니셜>/<실험명>/`에 저장하고, PR 본문에 실행 설정과 요약 점수를 기록합니다. 결과 이미지·보고서 파일은 별도 파일 공유 경로로 전달하며 Git에 추가하지 않습니다.

벤치마크 데이터와 산출물은 커밋 대상에서 제외합니다. 서브모듈에는 공식 저장소의 커밋 포인터만 기록합니다. [대용량 파일 관리](docs/git_guide.md)

기존 스냅샷과 입력 manifest는 과거 실험 재현의 기준입니다. 새로운 아이디어를 시험할 때 기존 결과를 먼저 덮어쓰지 마세요. [상세 Git 가이드](docs/git_guide.md)
