# 동결된 방법별 실행 코드

이 폴더에는 결과 이미지·로그·평가 보고서가 아닌 **실행 가능한 알고리즘 소스**만 보관합니다. 기존 `results/<방법>/stitching_snapshot.py`와 바이트 단위로 같은 코드입니다.

| 방법 | 코드 |
| --- | --- |
| Brown–Lowe 기반 baseline | [brown+lowe.py](brown+lowe.py) |
| RootSIFT | [RootSIFT.py](RootSIFT.py) |
| MAGSAC++ | [MAGSAC++.py](MAGSAC++.py) |
| RootSIFT + MAGSAC++ | [RootSIFT+MAGSAC++.py](RootSIFT+MAGSAC++.py) |
| 위 조합 + APAP | [RootSIFT+MAGSAC+++APAP.py](RootSIFT+MAGSAC+++APAP.py) |

```bash
python -B run.py --q all --code experiments/snapshots/RootSIFT+MAGSAC+++APAP.py --out results/local/apap
```

개발은 루트 `stitching.py`에서 진행하고 이 코드는 과거 방법의 재현 기준으로 유지합니다. 제출 시에는 선택한 코드를 `stitching.py` 이름의 단일 파일로 사용합니다.

외부 벤치마크·후처리 전 진단은 로컬 `results/<방법>/stitching_snapshot.py`가 있으면 해당 코드를, 없으면 이 폴더의 코드를 사용합니다. 평가 프로토콜에 실제 선택된 경로와 해시를 기록합니다. 새 코드 비교에는 `experiments/run_methods.py`로 로컬 실행 기록을 먼저 생성하세요.
