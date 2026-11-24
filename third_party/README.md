# 외부 벤치마크 안내·참조 코드

이 디렉터리는 공식 저장소를 Git submodule로 연결합니다. 상위 저장소는 각 저장소의 URL과 특정 커밋만 기록합니다.

| 경로 | 공식 저장소 | 용도 |
| --- | --- | --- |
| `third_party/hpatches-dataset` | [hpatches/hpatches-dataset](https://github.com/hpatches/hpatches-dataset) | HPatches 데이터·정답 H·배포 안내 |
| `third_party/udis-d` | [nie-lang/UnsupervisedDeepImageStitching](https://github.com/nie-lang/UnsupervisedDeepImageStitching) | UDIS-D 데이터 배포 안내와 원 논문 코드 |
| `third_party/udis2` | [nie-lang/UDIS2](https://github.com/nie-lang/UDIS2) | 마스크·PSNR·SSIM 평가 참조 코드 |

## 가져오기

```bash
# 새 clone
git clone --recurse-submodules https://github.com/jungejblue/IS_ws.git
cd IS_ws

# 기존 clone 또는 브랜치 변경 후
git submodule sync --recursive
git submodule update --init --recursive

# 실제 체크아웃한 커밋 확인
git submodule status --recursive
```

재현할 때는 `update --remote` 대신 위 명령으로 상위 저장소에 기록된 커밋을 사용합니다. 업데이트가 필요하면 해당 서브모듈에서 검토한 커밋을 체크아웃하고 상위 저장소에서 그 경로의 변경을 커밋합니다.

## 데이터와 실행 범위

서브모듈 초기화는 HPatches·UDIS-D 전체 이미지나 ZIP을 다운로드하지 않습니다. 실제 평가 데이터는 [데이터 준비 안내](../data/README.md)의 명령으로 `data/benchmarks/`에 저장하며 Git에서 제외합니다. 기존 로컬 데이터는 그대로 사용할 수 있습니다.

외부 저장소는 배포 안내·평가 코드 참고용입니다. 본 프로젝트의 과제 및 평가 실행기는 서브모듈 코드를 import하지 않으며, TensorFlow·PyTorch 모델을 실행하지 않습니다. 외부 저장소의 requirements나 environment.yml을 과제 환경에 설치할 필요가 없습니다. 외부 저장소에 포함된 테스트·예제·이미지는 원 저자의 구성으로 유지하며 상위 저장소에서는 개별 파일을 추적하지 않습니다.

Q1·Q2·Q3·E1·E2 정합은 서브모듈을 초기화하지 않아도 입력 사진과 프로젝트 의존성만 준비하면 실행할 수 있습니다. 평가기는 별도로 준비한 데이터와 본 프로젝트의 OpenCV·NumPy 코드를 사용합니다.
