"""Run four OpenCV/NumPy methods on the five assignment missions.

python3 -B experiments/run_methods.py
python3 -B experiments/run_methods.py --method rootsift_magsac_apap --q e2
Submission functions remain self-contained in stitching.py; run.py is unchanged.
"""
import argparse
import contextlib
import hashlib
import html
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

METHODS = {
    "rootsift": "RootSIFT",
    "magsac": "MAGSAC++",
    "rootsift_magsac": "RootSIFT+MAGSAC++",
    "rootsift_magsac_apap": "RootSIFT+MAGSAC+++APAP",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gallery():
    records = []
    for folder in METHODS.values():
        path = ROOT / "results" / folder / "experiment.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
    write_json(ROOT / "results" / "methods_summary.json", records)
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>네 방법의 정합 결과 비교</title>',
             '<style>body{font:16px/1.6 system-ui;margin:24px;color:#172536}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccd5df;padding:10px;vertical-align:top}img{width:100%;background:#111}small{display:block}a{color:#075cab}code{background:#eef3f8}p{max-width:1100px}</style>',
             '<h1>네 방법 × 다섯 미션</h1>',
             '<p>Q1·Q2·Q3·E1: 노출 보정·이음선·블렌딩 OFF. E2: 영상별 gain + 최대 가중치 마스크 + 5-band blending. 이미지를 누르면 원본 PNG가 열립니다.</p>',
             '<p>네 방법 모두 NumPy ray BA를 사용합니다. 기존 brown+lowe baseline의 OpenCV BA와 구현이 다릅니다. APAP는 공통 투영 좌표계 위 Moving DLT 보정의 과제용 확장으로 원 논문 BAMDLT의 완전 재현이 아닙니다.</p>',
             '<p>시간은 읽기·PNG 저장을 포함한 1회 실행입니다. 아래 결과는 HPatches/UDIS-D 평가가 아니며, fitting 진단 수치도 독립 대응점 정확도가 아닙니다.</p>',
             '<table><tr><th>미션</th>']
    for m in records:
        parts.append(f'<th>{html.escape(m["name"])}<small><a href="{m["folder"]}/experiment.json">설정·진단</a></small></th>')
    parts.append('</tr>')
    for q in course_runner.QUESTIONS:
        label = 'Q'+q if q.isdigit() else q.upper()
        parts.append(f'<tr><th>{label}</th>')
        for m in records:
            r = next((x for x in m['missions'] if x['question']==q), None)
            if r and r['status']=='ok':
                folder=m['folder']
                parts.append(f'<td><a href="{folder}/{label}_result.png"><img loading="lazy" src="{folder}/previews/{label}.jpg" alt="{label} {html.escape(m["name"])}"></a><small>{r["width"]}×{r["height"]} · {r["seconds"]}초</small><a href="{folder}/{label}_run.log">실행 로그</a></td>')
            else:
                parts.append('<td>미실행 또는 오류</td>')
        parts.append('</tr>')
    parts.append('</table><p>재실행: <code>python3 -B experiments/run_methods.py</code></p></html>')
    (ROOT/'results/comparison.html').write_text('\n'.join(parts),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['all',*METHODS], default='all')
    parser.add_argument('--q', choices=['all',*course_runner.QUESTIONS], default='all')
    args = parser.parse_args()
    source = ROOT/'stitching.py'
    source_hash = sha(source)
    # 기존 baseline은 재실행하거나 수정하지 않는다.
    baseline = ROOT/'results/brown+lowe'
    baseline_hashes = {str(p.relative_to(ROOT)):sha(p) for p in baseline.glob('*') if p.is_file()}
    methods = list(METHODS) if args.method=='all' else [args.method]
    questions = list(course_runner.QUESTIONS) if args.q=='all' else [args.q]
    failures = []
    for method in methods:
        out = ROOT/'results'/METHODS[method]
        (out/'previews').mkdir(parents=True,exist_ok=True)
        snapshot = out/'stitching_snapshot.py'
        snapshot.write_text(source.read_text(encoding='utf-8') +
            f'\nBASELINE_PROFILE = "brown_lowe_assignment"\nALGORITHM = {method!r}\n',encoding='utf-8')
        mod = course_runner.load_module(snapshot)
        manifest = dict(name=METHODS[method], algorithm=method, folder=out.name,
            created_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
            opencv_threads=cv2.getNumThreads(),source_sha256=source_hash,
            snapshot_sha256=sha(snapshot), runner_sha256=sha(ROOT/'run.py'),
            reproduce=f'python3 -B experiments/run_methods.py --method {method}',
            reproduce_frozen=f'python3 -B run.py --q all --code "results/{out.name}/stitching_snapshot.py" --out "results/{out.name}"',
            settings={k:v for k,v in vars(mod).items() if k.isupper() and isinstance(v,(str,int,float,bool,type(None)))},
            notes=[
                'Four methods use identical NumPy finite-difference LM ray BA; the preserved baseline used OpenCV ray BA. Solver implementation is an additional comparison variable versus the old baseline.',
                'MAGSAC++ recomputes H and inliers from all descriptor-match candidates; its selected inliers are used by NumPy BA and APAP, not native matcher inliers.',
                'Native detail matcher is used to obtain descriptor-match candidates; its preliminary RANSAC is superseded for MAGSAC methods.',
                'APAP variant: inverse Moving DLT on already globally warped images, shared correspondence-track targets, identity anchors, support fade, bounded correction, boundary fade and fold guard. Not exact APAP/BAMDLT paper reproduction.',
                'Q1 keeps the reference image fixed. Q2 composes on a plane before one cylindrical reprojection. Wide Q3/E1/E2 use one shared cylinder/sphere.',
                'Q1/Q2/Q3/E1 use hard overwrite without photometric postprocessing. E2 uses image-wide gain and max-weight multiband, no graph-cut.',
                'Timings are single executions including I/O, not repeated speed benchmarks.',
                'Fitting residuals are diagnostics, not held-out accuracy. No HPatches/UDIS-D benchmark evaluated in this run.',
            ], missions=[])
        for q in questions:
            label='Q'+q if q.isdigit() else q.upper()
            folder=course_runner.QUESTIONS[q][1]
            paths=sorted((p for p in (ROOT/'data'/folder).iterdir() if p.suffix.lower() in course_runner.IMG_EXT),key=course_runner.natural_key)
            mod.EXPERIMENT_DIAGNOSTICS.clear()
            with (out/f'{label}_run.log').open('w',encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    status=course_runner.run_one(mod,q,ROOT/'data'/folder,out)
            record=dict(status, diagnostics=list(mod.EXPERIMENT_DIAGNOSTICS),
                inputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],
                postprocessing='global_gain_max_weight_multiband' if q=='e2' else 'none')
            if status['status']=='ok':
                png=out/f'{label}_result.png'
                im=cv2.imread(str(png))
                if im is None or im.dtype!=np.uint8:
                    raise RuntimeError(f'Invalid PNG: {png}')
                record.update(width=im.shape[1],height=im.shape[0],output=png.name,output_sha256=sha(png))
                factor=min(1.,800/max(im.shape[:2]))
                preview=cv2.resize(im,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(out/'previews'/f'{label}.jpg'),preview):
                    raise RuntimeError('Preview write failed')
            else:
                failures.append((method,q,status.get('error',status['status'])))
            manifest['missions'].append(record)
            write_json(out/'experiment.json',manifest)
            print(f'{METHODS[method]} {label}: {status["status"]} ({status["seconds"]}s)',flush=True)
    for relative,digest in baseline_hashes.items():
        assert sha(ROOT/relative)==digest, f'Baseline changed: {relative}'
    assert sha(source)==source_hash, 'Source changed during execution'
    gallery()
    if failures:
        print('FAILURES:',failures)
        return 1
    print(f'Completed {len(methods)*len(questions)} missions; preserved baseline verified.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
