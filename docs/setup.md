# 설치와 환경 확인

## 벤치마크 참조 저장소

새 clone은 `git clone --recurse-submodules https://github.com/jungejblue/IS_ws.git`을 사용합니다. 기존 clone은 루트에서 `git submodule update --init --recursive`를 실행합니다. [서브모듈 안내](../third_party/README.md)를 참고하세요. 실제 벤치마크 데이터는 별도 다운로드하며 Q1~E2 정합만 실행할 때는 서브모듈이 필요하지 않습니다.

## 독립 가상환경

저장소 루트에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

`requirements.txt`는 OpenCV contrib의 headless 패키지를 사용합니다. 파일 입출력·SIFT·detail stitching·USAC를 사용하며 GUI 창은 필요하지 않습니다. `opencv-python`, `opencv-contrib-python`, 두 패키지의 headless 변형은 모두 `cv2`를 제공하므로 한 환경에 여러 종류를 중복 설치하지 않습니다.

이미 시스템 OpenCV가 정상 설치된 실험 장비에서 기존 환경을 그대로 재현할 때는 그 환경의 Python을 사용할 수 있습니다. 위 설치 명령으로 기존 시스템 환경을 덮어쓸 필요는 없습니다.

## 기록된 실험 환경

| 구성 | 측정 당시 버전 |
| --- | --- |
| Python | 3.10.12 |
| OpenCV | 4.5.4, 시스템 패키지 |
| NumPy | 1.26.4 |
| Matplotlib | 3.5.1 |

개발용 requirements는 이 환경과 비트 단위로 같은 결과를 보장하는 lockfile이 아닙니다. 특히 최종 PNG 평가기는 저장 결과의 정확한 재현을 요구하므로, 다른 OpenCV·BLAS 환경에서 실패하면 과거 점수와 새 환경의 결과를 섞지 말고 새 실험으로 기록합니다.

```bash
# 코드 실행에 필요한 API 확인
python scripts/check_environment.py

# 사진 이름·장수·디코딩 확인
python scripts/check_environment.py --data

# 게시된 결과에 사용한 사진과 해시까지 비교
python scripts/check_environment.py --data --reference-inputs
```

누락 사진은 [data/README.md](../data/README.md)에 따라 준비합니다. `--reference-inputs`는 기록된 Q3와 다른 새 촬영 세트에는 실패하는 것이 정상입니다. 새 사진은 별도 실험과 manifest로 관리합니다.

## 흔한 문제

| 현상 | 확인할 사항 |
| --- | --- |
| `No module named cv2` | 활성 가상환경에서 requirements 설치 여부 |
| SIFT·USAC·detail API 누락 | 설치한 OpenCV 배포판·버전과 중복 설치 여부 |
| `Expected ... image paths` | 입력 장수, 확장자, 파일명 번호 순서 |
| 원통·구면 결과가 예상과 다름 | 루트 코드인지 스냅샷인지, 프로필과 입력 manifest |
| 저장 PNG 재현 검사 실패 | 코드·원본·OpenCV/NumPy/BLAS 환경 변경 여부 |
| Matplotlib 캐시 경고 | 필요하면 `MPLCONFIGDIR=/tmp/is-ws-matplotlib` 설정 |

브랜치에서 의존성을 바꾸면 버전과 설치 방법, 기존 테스트 결과를 PR에 기록합니다.
