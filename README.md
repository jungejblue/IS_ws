# IS_ws — OpenCV Image Stitching

OpenCV와 NumPy로 이미지 정합을 구현하고 SIFT·AKAZE, RootSIFT, RANSAC·MAGSAC++, APAP·REW·DIS를 비교합니다. 알고리즘, 실험 케이스 선택, 벤치마크 및 미션 평가는 `stitching.py` 한 파일에 있습니다. `cv2.Stitcher`는 사용하지 않습니다.

## 저장소 구성

```text
IS_ws/
├── .gitattributes
├── .gitignore
├── Agents.md                     # 저장소 작업 규칙
├── README.md
├── requirements.txt
├── run.py                        # 과제 제공 실행기, 수정하지 않음
├── stitching.py                  # 알고리즘 + 직접 실행 CLI
├── data/
│   ├── Q1/, Q2/, Q3/             # 제공·직접 촬영 입력 사진 13장
│   ├── README.md
│   └── benchmark_sources.json    # 외부 데이터 출처·ZIP 해시
├── docs/
│   ├── algorithm.md              # 처리 흐름·방법론·33케이스 정의
│   ├── benchmarks.md             # 데이터셋·평가 지표·해석
│   ├── benchmark_results.md      # 외부 벤치마크 측정 기록
│   ├── mission_results.md        # Q1~E2 측정 기록
│   └── assignment/               # 프로젝트 공지 PDF
└── third_party/
    └── README.md                 # 로컬 참조 저장소 안내
```

원격에는 벤치마크 저장소, 데이터 ZIP, 압축 해제 벤치마크 데이터, 결과 이미지가 포함되지 않습니다. Git submodule도 사용하지 않습니다. 아래 명령으로 필요한 참조 저장소와 데이터를 로컬에 준비합니다. 저장소 작업 시 [Agents.md](Agents.md)를 먼저 확인하세요.

## 설치

Python 3.10 이상, CPU 환경을 사용합니다.

```bash
git clone https://github.com/jungejblue/IS_ws.git
cd IS_ws
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 측정 재현에 사용할 스레드 설정
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

기록된 결과의 환경은 Python 3.10.12 / OpenCV 4.5.4 / NumPy 1.26.4입니다. `requirements.txt`는 설치 가능한 개발 버전 범위이며 완전히 동일한 환경의 잠금 파일은 아닙니다. 버전·입력·선택한 방법 수가 바뀌면 과거 수치와 달라질 수 있습니다. OpenCV wheel은 한 종류만 설치합니다.

## 이미지 정합 실행

```bash
# 기본: 33방법 × Q1·Q2·Q3·E1·E2 = 최종 PNG 165장
python -B stitching.py

# 전체 케이스 확인
python -B stitching.py --task list

# DIS 8조합의 다섯 미션: PNG 40장
python -B stitching.py --method dis --workers 2

# 특정 케이스의 Q2만
python -B stitching.py --case D03 --mission Q2

# 같은 상위 조건에서 국소 보정 없음/APAP/REW/DIS 비교
python -B stitching.py --case F06 F07 R03 D03 --mission Q2

# 다른 입력·출력 디렉터리
python -B stitching.py --case D03 --data-root /path/to/data --output /path/to/results
```

`--method`는 `brown_lowe`, `global`, `apap`, `rew`, `dis`, `all`을 받습니다. `global`은 국소 보정 없는 8조합입니다. `--case`에는 B0, F00~F15, R00~R07, D00~D07 또는 전체 ID를 지정합니다. 두 필터를 함께 지정하면 교집합을 실행합니다. `--mission`은 Q1, Q2, Q3, E1, E2, all이며 기본값은 all입니다.

```text
results/                         # 실행할 때 생성, Git 제외
├── brown_lowe/B0_reference/
│   ├── Q1_result.png
│   ├── Q2_result.png
│   ├── Q3_result.png
│   ├── E1_result.png
│   └── E2_result.png
├── global/F06_sift_rootsift_magsac/
├── apap/F07_sift_rootsift_magsac_apap/
├── rew/R03_sift_rootsift_magsac_rew/
└── dis/D03_sift_rootsift_magsac_dis/
    ├── Q1_result.png
    ├── Q2_result.png
    ├── Q3_result.png
    ├── E1_result.png
    └── E2_result.png
```

각 선택된 케이스 폴더에는 최종 PNG만 저장합니다. JSON·로그·미리보기·스냅샷·평가 캐시를 남기지 않습니다. 같은 케이스를 다시 실행하면 선택한 PNG를 갱신하며, 실패한 재실행에 과거 PNG가 남아 성공으로 오인되지 않도록 해당 파일을 제거합니다. 실패는 콘솔과 종료 코드로 확인합니다.

기본 병렬 작업자는 2개이며 메모리에 여유가 있으면 `--workers 4`를 사용할 수 있습니다. `--verbose`로 알고리즘 단계 로그를 콘솔에 표시합니다. 기존 미션 전체 결과를 모두 비교하려면 일부 입력만 선택하지 말고 다섯 미션을 함께 실행하세요.

## 과제 실행기와의 차이

`q1(img_paths)`부터 `e2(img_paths)`까지는 BGR `uint8` 이미지 한 장만 반환하며 파일을 저장하지 않습니다. 직접 실행 CLI가 모든 케이스의 저장을 담당합니다. `run.py`는 원본 그대로 유지하며 다음과 같이 제출 인터페이스를 검사할 수 있습니다.

```bash
python -B run.py --q 2 --out /tmp/is-ws-submission-check
```

제공된 `run.py`는 PNG 외에 상태 JSON도 생성하므로 PNG 전용 결과 폴더를 만들 때는 `stitching.py`를 직접 실행하세요. `run.py`는 루트 함수 기본 설정을 사용하며, 33케이스 CLI는 비교용 프로필을 명시합니다. 비교 프로필은 Q1~E1에서 후처리 없이 덮어쓰기하고, E2만 gain·최대 가중치 마스크·멀티밴드 합성을 적용합니다.

## 벤치마크 참조 저장소를 로컬에 clone

새 환경에서 실행합니다. 이미 clone한 디렉터리에는 다시 clone하지 않습니다. 이 디렉터리들은 `.gitignore`로 제외되어 원격에 올라가지 않습니다.

```bash
git clone https://github.com/hpatches/hpatches-dataset.git third_party/hpatches-dataset
git clone https://github.com/nie-lang/UnsupervisedDeepImageStitching.git third_party/udis-d
git clone https://github.com/nie-lang/UDIS2.git third_party/udis2

# 본 프로젝트에서 확인한 참조 버전으로 고정
git -C third_party/hpatches-dataset checkout --detach fd79bca4dc755c9f623a5f129f3452fc663ee305
git -C third_party/udis-d checkout --detach 11ce1eb0c25e0e888533fb82ba16363aae375d05
git -C third_party/udis2 checkout --detach b4158950ab1411268c67a48169ba732a19d399d0
```

이 저장소들은 데이터 배포 안내와 평가 방법 참고용입니다. clone만으로 실제 데이터가 다운로드되지는 않습니다. 외부 저장소의 `requirements.txt`를 설치하거나 TensorFlow·PyTorch 모델을 실행할 필요는 없습니다. 본 프로젝트의 정합·평가 코드는 이 저장소들을 import하지 않습니다.

## 벤치마크 데이터 다운로드·검증·압축 해제

실제 이미지는 다음 위치에 준비합니다.

```text
data/benchmarks/                       # 전체 Git 제외
├── hpatches-sequences-release/
│   ├── i_ajuntament/
│   │   ├── 1.ppm ... 6.ppm
│   │   └── H_1_2 ... H_1_6
│   └── ...                           # 총 116개 시퀀스
└── testing/
    ├── input1/*.jpg
    └── input2/*.jpg
```

아래 명령은 기록된 배포본을 내려받고 SHA-256을 검사한 후 압축을 풉니다. HPatches URL은 기록된 미러이며 공식 안내는 [HPatches](https://github.com/hpatches/hpatches-dataset), UDIS-D 안내는 [공식 저장소](https://github.com/nie-lang/UnsupervisedDeepImageStitching)에 있습니다. URL과 해시는 [data/benchmark_sources.json](data/benchmark_sources.json)에도 보존합니다.

```bash
mkdir -p data/benchmarks

curl -fL --retry 3 \
  'https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip?download=true' \
  -o data/benchmarks/hpatches-sequences-release.zip

curl -fL --retry 3 \
  'https://drive.usercontent.google.com/download?id=1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0&export=download&confirm=t' \
  -o data/benchmarks/udis_testing.zip

# 해시 검사 성공 후에만 압축 해제
python - <<'PY'
import hashlib, json, pathlib, stat, zipfile
base = pathlib.Path('data/benchmarks').resolve()
metadata = json.loads(pathlib.Path('data/benchmark_sources.json').read_text())
for name, expected_root in [('hpatches-sequences-release.zip', 'hpatches-sequences-release'),
                            ('udis_testing.zip', 'testing')]:
    path = base / name
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != metadata['downloads'][name]['sha256']:
        raise SystemExit('SHA-256 mismatch: ' + name)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            relative = pathlib.PurePosixPath(item.filename)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in item.filename
                    or ':' in item.filename or not relative.parts
                    or relative.parts[0] != expected_root
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise SystemExit('Unexpected archive path: ' + item.filename)
        archive.extractall(base)
    print('Verified and extracted:', name)
PY

python -B stitching.py --task check-data --decode

# 검증 후 ZIP은 삭제하고 압축 해제 데이터만 유지
rm data/benchmarks/hpatches-sequences-release.zip data/benchmarks/udis_testing.zip
```

Google Drive가 HTML 확인 화면을 내려주면 해시 검사가 실패합니다. 이 경우 공식 배포 안내에서 시험 ZIP을 내려받아 같은 경로에 둔 뒤 다시 검사하세요. 해시가 다른 배포본을 기존 결과와 같은 데이터라고 간주하지 않습니다. 기록된 UDIS-D 배포본에는 `input1/000001.jpg`가 없어 완전한 영상 쌍은 1,105개이며, 이를 임의로 채우지 않습니다.

## 벤치마크·미션 평가

```bash
# 외부 1,685쌍 × 33방법: docs/benchmark_results.md 갱신
python -B stitching.py --task benchmark --workers 4

# 각 데이터셋 1쌍으로 시험, 보존된 문서는 덮어쓰지 않음
python -B stitching.py --task benchmark --limit 1 --report /tmp/is-ws-benchmark-pilot.md

# 선택한 미션을 생성·평가: docs/mission_results.md 갱신
python -B stitching.py --task evaluate --workers 2

# 특정 케이스만 별도 문서로 평가
python -B stitching.py --task evaluate --case F06 F07 R03 D03 \
  --mission Q2 --report /tmp/is-ws-q2-evaluation.md
```

평가 명령은 Markdown 문서만 추가로 저장합니다. `results/`는 PNG 전용입니다. 미션 평가 중 필요한 좌표 캐시는 시스템 임시 디렉터리에 생성하고 종료 시 제거합니다. 외부 평가의 원시 JSON·CSV·중간 체크포인트는 저장하지 않으므로 중단되면 다시 실행해야 합니다. 결과 문서를 보존하려면 `--report`로 다른 `.md` 경로를 선택하세요.

외부 평가는 Q1 두 영상 경로, 미션 평가는 저장된 최종 PNG의 원본 일치도입니다. HPatches 정답 기하와 UDIS-D·미션 PSNR/SSIM을 구분해서 해석해야 합니다. 비교 방법 수가 달라지면 공통 영역도 달라집니다.

## 문서

- [알고리즘과 방법론](docs/algorithm.md)
- [벤치마크와 평가 지표](docs/benchmarks.md)
- [외부 벤치마크 결과](docs/benchmark_results.md)
- [프로젝트 미션 결과](docs/mission_results.md)
- [HDY의 Q1·Q2·Q3·E1·E2 결과](docs/hdy_mission_results.md)
- [HDY의 8방법 비교 결과](docs/hdy_method_comparison.md)
- [프로젝트 공지](docs/assignment/이미지스티칭_프로젝트_안내.pdf)

`stitching_hdy.py`는 HDY의 미션별 선택 코드입니다(Q1 Brown–Lowe, Q2 RootSIFT·MAGSAC, Q3·E1·E2 APAP식 국소 보정). `run_hdy.py --q all`로 확인할 수 있습니다. HDY의 완성 사진 5장 ZIP은 Git 밖에서 별도 공유합니다. 외부 벤치마크를 이 코드로 측정한 결과는 없습니다.

생성 결과, 압축 파일, 벤치마크 clone 및 데이터는 강제로 Git에 추가하지 않습니다. 변경한 코드의 검증 산출물도 작업이 끝나면 제거하고, 유지할 측정값은 `docs/`의 결과 Markdown 문서에 정리합니다.
