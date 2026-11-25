# 로컬 벤치마크 참조 저장소

이 디렉터리에서는 이 README만 Git으로 관리합니다. 아래 저장소의 clone은 로컬에서만 유지하며 submodule을 사용하지 않습니다.

| 로컬 경로 | 공식 저장소 | 용도 |
| --- | --- | --- |
| `hpatches-dataset/` | [hpatches/hpatches-dataset](https://github.com/hpatches/hpatches-dataset) | 데이터·정답 H·배포 안내 |
| `udis-d/` | [nie-lang/UnsupervisedDeepImageStitching](https://github.com/nie-lang/UnsupervisedDeepImageStitching) | UDIS-D 데이터 배포 안내 |
| `udis2/` | [nie-lang/UDIS2](https://github.com/nie-lang/UDIS2) | 광도 평가 참조 코드 |

clone·버전 고정·데이터 다운로드·압축 해제 명령은 [프로젝트 README](../README.md)에 있습니다. 저장소 clone만으로 실제 이미지가 준비되지는 않습니다. 데이터는 `data/benchmarks/`에 별도로 준비합니다.

프로젝트 코드는 이 저장소들을 import하지 않으며 외부 TensorFlow·PyTorch 모델을 실행하지 않습니다. clone 내부의 환경 설정이나 requirements를 프로젝트 환경에 설치할 필요가 없습니다.
