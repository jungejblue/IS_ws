# Git·PR 협업 방법

## 개인 브랜치

작업 전 변경 사항을 확인한 뒤 최신 `main`에서 본인 이니셜과 작업명을 포함한 브랜치를 만듭니다.

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c jj/fix-matching
```

`jj`를 본인 이니셜로 바꿉니다. 한 브랜치에는 하나의 설명 가능한 작업을 담습니다. 코드에 필요한 의존성 변경은 requirements와 설치 문서를 함께 수정합니다.

## 검증·커밋·PR

```bash
python scripts/check_environment.py
python scripts/check_repository.py
git diff --check
git diff --stat

# 실제 수정한 파일만 선택
git add README.md docs/notes.md
git commit -m "docs: clarify stitching experiment workflow"
git push -u origin jj/fix-matching
```

GitHub에서 `main`을 대상으로 PR을 열고 다른 팀원이 검토한 뒤 병합합니다. PR에는 바꾼 동작·이유, 실행 명령·입력, 확인한 내용, 결과 차이와 한계를 적습니다. 저장소 검사는 실험 성능을 대신하지 않으므로 기하·합성 변경은 실제 미션 결과도 확인합니다.

## 데이터와 결과 공유

- 과제 원본과 외부 벤치마크는 `data/`에 보관하되 Git에 넣지 않습니다. Q3 사진은 팀 내부에서 전달합니다.
- 개인 실행은 `results/local/<이니셜>/<실험명>/`에 저장합니다.
- PR 본문에는 설정·입력 해시·요약 점수를 적고, 이미지·보고서는 팀의 별도 파일 공유 경로로 전달합니다.
- `results/` 전체는 Git 제외 대상입니다. 미리보기·CSV·JSON·HTML·PDF도 포함됩니다.
- 과거 방법의 실행 소스는 `experiments/snapshots/`에서 관리합니다.
- 테스트 코드와 테스트용 CI 워크플로는 삭제했습니다. 무시 규칙으로 재추적을 방지합니다.
- Python 캐시·가상환경·테스트 캐시·coverage·로그·임시 파일·개인 설정도 Git에서 제외합니다.

## 기존 대용량 파일 정리의 의미

이번 정리는 이미 추적하던 데이터·모든 정합 및 평가 산출물의 **Git 추적만 해제**하고 로컬 파일은 유지합니다. 커밋 화면에서는 삭제로 표시되지만 작업 폴더에서 데이터를 지우는 작업이 아닙니다.

이미 원격에 올라간 과거 커밋의 대용량 blob은 새 `.gitignore`나 추적 해제로 없어지지 않습니다. 이 작업에서는 과거 이력을 재작성하거나 강제 push하지 않습니다. 전체 이력 용량 축소는 팀과 조율할 별도 작업입니다.

이 정리 커밋을 받는 기존 팀원은 `data/`와 재생성하기 어려운 로컬 출력의 사본을 저장소 밖에 먼저 확보하세요. Git에서 추적 파일이 삭제되는 커밋을 체크아웃하면 그 팀원의 작업 트리에서는 해당 파일이 제거될 수 있습니다. 이후 같은 위치에 사본을 돌려놓으면 무시 규칙의 적용을 받습니다.

`.gitignore`만 추가했다고 이미 추적 중인 파일이 제외되는 것은 아닙니다. `python scripts/check_repository.py`가 추적 중인 로컬 산출물이 남았는지 확인합니다.

## 서브모듈 관리

`third_party/`의 공식 저장소는 커밋 포인터로 관리합니다. clone 시 `--recurse-submodules`를 사용하고, 기존 clone에서는 `git submodule sync --recursive`와 `git submodule update --init --recursive`를 실행합니다. [상세 안내](../third_party/README.md)

실험 재현 시 임의로 `git submodule update --remote`를 실행하지 않습니다. 의도적으로 버전을 변경했다면 해당 경로를 `git add`하여 상위 저장소에 새 커밋 포인터를 기록하고 변경 이유를 PR에 적습니다. 데이터와 생성 결과는 서브모듈 디렉터리에 저장하지 않습니다.

## 미전송 이력의 대용량 파일 정리

이전 로컬 커밋 `11b77b1`에 벤치마크 ZIP이 포함되어 있었고, 후속 `c0ad1b7`에서 추적을 해제했습니다. 삭제 커밋만 추가하면 앞선 커밋의 대용량 파일도 push 대상에 남으므로, 원격 main `1fb0660` 위에 정리된 현재 상태를 `arrange_complete` 메시지의 새 커밋 하나로 합쳤습니다. 서브모듈·문서 변경도 함께 포함합니다.

원래 이력은 로컬 `backup/before-artifact-cleanup-c0ad1b7` 브랜치에 보관합니다. 이 백업에는 과거 대용량 파일이 남아 있으므로 원격으로 보내지 말고 `main`만 push합니다. 로컬 데이터와 정합 산출물은 삭제하지 않습니다. 원격에 이미 존재하는 과거 이력은 이번 정리 대상이 아닙니다.
