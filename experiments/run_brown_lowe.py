"""Generate the assignment baseline with the unchanged course runner.

Run from any directory: python3 experiments/run_brown_lowe.py
The submitted implementation remains self-contained in stitching.py.
"""

import contextlib
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as course_runner


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "results" / "brown+lowe"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "stitching.py"
    snapshot = out / "stitching_snapshot.py"
    snapshot.write_text(source.read_text(encoding="utf-8") +
                        '\nBASELINE_PROFILE = "brown_lowe_assignment"\n', encoding="utf-8")
    mod = course_runner.load_module(snapshot)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": "Brown-Lowe-based assignment baseline",
        "profile": mod.BASELINE_PROFILE,
        "exact_paper_reproduction": False,
        "reference": "https://www.cs.ubc.ca/~lowe/papers/07brown.pdf",
        "reproduce": "python3 experiments/run_brown_lowe.py",
        "reproduce_frozen": "python3 run.py --q all --code results/brown+lowe/stitching_snapshot.py --out results/brown+lowe",
        "python": platform.python_version(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "source_sha256": digest(source), "snapshot_sha256": digest(snapshot),
        "runner_sha256": digest(ROOT / "run.py"),
        "settings": {k: v for k, v in vars(mod).items()
                     if k.isupper() and isinstance(v, (str, int, float, bool, type(None)))},
        "adaptations": [
            "Q1 uses a two-image planar homography without bundle adjustment.",
            "Q2/3/E1/E2 use OpenCV homography initialization, ray BA and wave correction; not the paper's exact optimizer.",
            "Q3/E1/E2 use known 2x4 adjacency and keep all eight inputs; no unordered panorama recognition.",
            "Translation shortcut disabled for this baseline.",
            "Q2 composes on a plane then reprojects once to a cylinder.",
            "Wide Q3/E1/E2 use a shared cylinder/sphere because a single forward plane cannot contain all inputs.",
            "Q1/Q2/Q3/E1 omit exposure, seam finding and blending per user-selected assignment protocol.",
            "E2 uses OpenCV image-wide gain, warped-mask-distance max-weight ownership and 5-band blending.",
            "E2 weight initialization approximates the paper; no graph-cut or block gain.",
        ],
        "measurement_notes": [
            "Single execution per mission; elapsed time includes image I/O and PNG encoding.",
            "Alignment logs use fitting inliers, not independent ground-truth accuracy.",
            "No PSNR/SSIM/held-out accuracy measured in this generation run.",
        ],
        "missions": [],
    }
    for q, (_, folder, expected) in course_runner.QUESTIONS.items():
        label = f"Q{q}" if q.isdigit() else q.upper()
        paths = sorted((p for p in (ROOT / "data" / folder).iterdir()
                        if p.suffix.lower() in course_runner.IMG_EXT),
                       key=course_runner.natural_key)
        log_path = out / f"{label}_run.log"
        with log_path.open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                status = course_runner.run_one(mod, q, ROOT / "data" / folder, out)
        record = dict(status)
        record["inputs"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
        record["postprocessing"] = "global_gain_max_weight_multiband" if q == "e2" else "none"
        record["log"] = log_path.name
        if status["status"] == "ok":
            png = out / f"{label}_result.png"
            image = cv2.imread(str(png))
            if image is None or image.dtype != np.uint8 or len(paths) != expected:
                raise RuntimeError(f"Invalid saved result: {label}")
            record.update(width=image.shape[1], height=image.shape[0],
                          output=png.name, output_sha256=digest(png))
        manifest["missions"].append(record)
        print(f"{label}: {status['status']}, {status['seconds']}s -> {out / (label + '_result.png')}", flush=True)
    (out / "experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["status"] == "ok" for r in manifest["missions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
