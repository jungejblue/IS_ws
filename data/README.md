# 입력 데이터

원본 사진과 외부 벤치마크는 로컬에 보관하며 Git에서는 이 문서, [입력 manifest](input_manifest.json), [벤치마크 배포 설정](benchmark_sources.json)만 관리합니다.

```text
data/
├─ Q1/Q1_img_1.jpg, Q1_img_2.jpg
├─ Q2/Q2_img_1.JPG ... Q2_img_3.JPG
├─ Q3/Q3_img_1.jpeg ... Q3_img_8.jpeg
└─ benchmarks/
   ├─ hpatches-sequences-release/
   └─ testing/input1/, testing/input2/
```

Q1/Q2는 과제에서 제공한 사진을, Q3는 팀이 촬영한 사진을 팀 내부 전달 경로로 받아 넣습니다. 기존 결과를 재현하려면 manifest와 같은 원본이 필요합니다. 새로운 Q3 사진은 해시가 달라지므로 별도 실험으로 취급합니다.

```bash
python scripts/check_environment.py --data
python scripts/check_environment.py --data --reference-inputs
```

Q3는 2×4 raster order, 윗줄 왼쪽부터 1~4, 아랫줄 5~8입니다. `.jpg`, `.jpeg`, `.png`, `.bmp`는 실행기가 지원합니다. 과제 제출본 파일명은 `.jpg`로 맞춥니다.

## 외부 벤치마크 준비

공식 안내 저장소는 `third_party/hpatches-dataset`과 `third_party/udis-d`, 평가 참고 코드는 `third_party/udis2` 서브모듈로 연결합니다. [서브모듈 초기화 안내](../third_party/README.md)에 따라 가져올 수 있습니다.

서브모듈 초기화와 실제 데이터 다운로드는 별도입니다. 전체 이미지·ZIP은 아래 명령으로 받고, 상위 저장소와 서브모듈 어디에도 추가하지 않습니다. 기존 `data/benchmarks/` 경로와 해시 검증 방식은 유지합니다.

```bash
# 원본 압축 다운로드; 네트워크와 수 GB의 디스크 공간 필요
python experiments/download_benchmarks.py hpatches
python experiments/download_benchmarks.py drive-file \
  --id 1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0 --name udis_testing.zip

# 기록된 SHA-256 확인 후 안전하게 압축 해제
python scripts/prepare_benchmarks.py
```

다운로드 스크립트는 압축을 받으며, 해제는 별도 명령입니다. `prepare_benchmarks.py`는 [배포처·해시 설정](benchmark_sources.json)의 해시와 전체 파일 목록·내용을 확인합니다. 이미 해제된 정상 파일은 다시 쓰지 않으며 충돌 파일은 덮어쓰지 않습니다.

전체 HPatches 116개 시퀀스의 580쌍을 사용했습니다. UDIS-D 테스트 압축은 `input1/000001.jpg`가 없어 완전한 1,105쌍을 사용했습니다. 반드시 **같은 파일명**으로 짝을 구성합니다. 배포 내용이 변경되어 해시가 다르면 기존 평가를 그대로 재현하는 데이터로 취급하지 말고 출처·누락 상태를 다시 기록합니다.
