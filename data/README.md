# 입력 사진과 로컬 벤치마크

Q1은 사진 2장, Q2는 3장, Q3는 8장을 사용하며 E1/E2는 Q3 입력을 재사용합니다. Q1/Q2/Q3 입력 사진 13장은 저장소에 포함하고 파일명의 숫자 순서대로 읽습니다. Q3 과제 촬영 조건은 2×4 격자, 이웃 사이 20~30% 중첩, 윗줄 왼쪽부터 raster 순서입니다.

`data/benchmarks/`는 로컬 전용입니다. HPatches의 `hpatches-sequences-release/`와 UDIS-D의 `testing/input1/`, `testing/input2/`를 압축 해제된 상태로 둡니다. ZIP·원시 벤치마크 데이터는 Git에 올리지 않습니다.

[프로젝트 README](../README.md)에 참조 저장소 clone과 데이터 준비 명령이 있습니다. [benchmark_sources.json](benchmark_sources.json)은 사용한 배포본의 다운로드 URL·파일 크기·SHA-256을 기록합니다. UDIS-D의 해당 배포본은 `input1/000001.jpg`가 없어 완전한 시험 영상 쌍이 1,105개입니다.

```bash
python -B stitching.py --task check-data --decode
```

원본 과제 사진이나 외부 데이터가 달라지면 과거 결과 문서와 수치가 같다고 가정하지 않습니다.
