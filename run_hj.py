"""
실행/검사 스크립트 (조교 제공 run.py 기반, stitching_hj.py 실행 + 출력은 [OK] 만)

사용법
  python run.py --q 1                 # data/Q1 이미지로 q1() 실행 → results/Q1_result.png
  python run.py --q 2
  python run.py --q 3                 # data/Q3 에 직접 찍은 8장을 넣어두세요
  python run.py --q e1
  python run.py --q all               # 전부 실행
  python run.py --q 1 --input 다른폴더 --out 다른폴더

결과 파일 (고정 파일명)
  results/Q1_result.png               정합 결과 (Q2_result.png, ..., E2_result.png)
  results/Q1_status.json              실행 성공 여부, 에러 메시지, 실행 시간
"""

import argparse
import importlib.util
import json
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

QUESTIONS = {
    "1": ("q1", "Q1", 2),
    "2": ("q2", "Q2", 3),
    "3": ("q3", "Q3", 8),
    "e1": ("e1", "Q3", 8),
    "e2": ("e2", "Q3", 8),
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_module(code_path):
    code_path = Path(code_path).resolve()
    spec = importlib.util.spec_from_file_location("stitching", code_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(code_path.parent))
    before = set(sys.modules)
    spec.loader.exec_module(mod)
    # 제출은 stitching.py 하나뿐이므로, 직접 만든 다른 .py를 import하면 채점 때 실패한다
    local = sorted(
        Path(sys.modules[name].__file__).resolve().relative_to(code_path.parent).as_posix()
        for name in set(sys.modules) - before
        if getattr(sys.modules[name], "__file__", None)
        and code_path.parent in Path(sys.modules[name].__file__).resolve().parents
    )
    if local:
        sys.exit(f"[ERROR] stitching.py가 직접 만든 다른 파일을 import합니다: {', '.join(local)}\n"
                 f"        모든 코드는 stitching.py 한 파일 안에 작성하세요.")
    return mod


def check_image(img):
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or img.size == 0:
        raise TypeError(f"반환값은 numpy 이미지 배열이어야 합니다 (type={type(img).__name__})")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def run_one(mod, q, input_dir, out_dir):
    func_name, _, n_expected = QUESTIONS[q]
    label = f"Q{q}" if q.isdigit() else q.upper()
    result_path = out_dir / f"{label}_result.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"question": q, "func": func_name, "input": str(input_dir)}

    paths = sorted((p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXT), key=natural_key)
    status["n_images"] = len(paths)
    if len(paths) != n_expected:
        print(f"[경고] {func_name}: 이미지 {n_expected}장을 기대했지만 {len(paths)}장이 있습니다 ({input_dir})")

    t0 = time.time()
    try:
        result = check_image(getattr(mod, func_name)([str(p) for p in paths]))
        cv2.imwrite(str(result_path), result)
        status["status"] = "ok"
        print("[OK]")
    except NotImplementedError:
        status["status"] = "not_implemented"
        print(f"[--] {func_name}: 구현되지 않음")
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
        print(f"[ERROR] {func_name}: {status['error']}")
    status["seconds"] = round(time.time() - t0, 2)
    (out_dir / f"{label}_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def main():
    ap = argparse.ArgumentParser(description="Image stitching 과제 실행기")
    ap.add_argument("--q", required=True, choices=list(QUESTIONS) + ["all"])
    ap.add_argument("--input", help="입력 이미지 폴더 (기본: data/Q1, data/Q2, data/Q3)")
    ap.add_argument("--out", default="results", help="결과 저장 폴더 (기본: results)")
    ap.add_argument("--code", default=str(Path(__file__).parent / "stitching_hj.py"), help="stitching 코드 경로")
    args = ap.parse_args()

    base = Path(__file__).parent
    mod = load_module(args.code)
    qs = list(QUESTIONS) if args.q == "all" else [args.q]
    if args.input and len(qs) > 1:
        ap.error("--q all 과 --input 은 같이 쓸 수 없습니다")

    results = []
    for q in qs:
        data_folder = QUESTIONS[q][1]
        input_dir = Path(args.input) if args.input else base / "data" / data_folder
        results.append(run_one(mod, q, input_dir, Path(args.out)))
    return 0 if all(r["status"] != "error" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
