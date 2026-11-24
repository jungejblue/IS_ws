"""Evaluate frozen five-method Q1 pairwise pipelines using OpenCV/NumPy only.

No ground-truth homography is used for fitting. No result PNG is modified.
python3 -B experiments/evaluate_benchmarks.py --workers 4
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/benchmarks'
OUT = ROOT/'results/evaluation'
FOLDERS = ['brown+lowe','RootSIFT','MAGSAC++','RootSIFT+MAGSAC++','RootSIFT+MAGSAC+++APAP']
MODULES = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_code(folder):
    """Use a local run's snapshot, or the checked-in method source on a fresh clone."""
    local = ROOT/'results'/folder/'stitching_snapshot.py'
    return local if local.is_file() else ROOT/'experiments/snapshots'/f'{folder}.py'


def load_modules():
    global MODULES
    cv2.setNumThreads(1)
    MODULES = []
    for i,folder in enumerate(FOLDERS):
        path=method_code(folder)
        spec=importlib.util.spec_from_file_location('frozen_method_'+str(i),path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.VERBOSE=False
        MODULES.append(mod)


def project(points, h):
    p=np.column_stack([points,np.ones(len(points))])@h.T
    with np.errstate(divide='ignore',invalid='ignore'):
        return p[:,:2]/p[:,2,None]


def ssim_map(a,b,gaussian=True):
    a,b=a.astype(np.float64),b.astype(np.float64)
    if gaussian:
        blur=lambda x: cv2.GaussianBlur(x,(11,11),1.5,borderType=cv2.BORDER_REFLECT)
        covariance_factor=1.0
    else:
        blur=lambda x: cv2.blur(x,(7,7),borderType=cv2.BORDER_REFLECT)
        covariance_factor=49/48
    ma,mb=blur(a),blur(b)
    va=covariance_factor*(blur(a*a)-ma*ma)
    vb=covariance_factor*(blur(b*b)-mb*mb)
    cov=covariance_factor*(blur(a*b)-ma*mb)
    score=((2*ma*mb+2.55**2)*(2*cov+7.65**2))/((ma*ma+mb*mb+2.55**2)*(va+vb+7.65**2))
    return score.mean(axis=2)


def psnr(a,b,mask):
    if not np.any(mask):
        return None
    diff=a[mask].astype(np.float64)-b[mask].astype(np.float64)
    mse=float(np.mean(diff*diff))
    return min(100.0,10*math.log10(255.**2/max(mse,1e-12)))


def safe_mask(mask,window=11):
    return cv2.erode(mask.astype(np.uint8),np.ones((window,window),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0)>0


def pair_warp(mod,images,paths,grid=None,hgt=None):
    """Capture the exact frozen Q1 geometry/layers before hard composition."""
    capture={}
    originals={name:getattr(mod,name) for name in ['_read_images','_working_images','_brown_lowe_compose']}
    original_find=cv2.findHomography
    original_field=getattr(mod,'_moving_dlt_field',None)
    def read(_paths,expected):
        assert expected==2
        return images
    def work(*args,**kwargs):
        result=originals['_working_images'](*args,**kwargs)
        capture['scales']=result[1]
        return result
    def find(*args,**kwargs):
        result=original_find(*args,**kwargs)
        capture['h_work']=result[0]
        capture['fit_inliers']=0 if result[1] is None else int(np.count_nonzero(result[1]))
        return result
    def compose(warped,masks,corners,**kwargs):
        capture['warped']=warped;capture['masks']=masks
        return np.zeros((1,1,3),np.uint8),np.ones((1,1),np.float32),(0,0)
    def field(*args,**kwargs):
        result=original_field(*args,**kwargs)
        capture['field']=result[0]
        return result
    mod._read_images=read;mod._working_images=work;mod._brown_lowe_compose=compose
    cv2.findHomography=find
    if original_field:
        mod._moving_dlt_field=field
    start=time.perf_counter()
    try:
        if hasattr(mod,'EXPERIMENT_DIAGNOSTICS'):
            mod.EXPERIMENT_DIAGNOSTICS.clear()
        mod.q1(paths)
        s0,s1=capture['scales']
        h21=np.linalg.inv(s0)@capture['h_work']@s1
        h01=np.linalg.inv(h21)
        h,w=images[0].shape[:2];ht,wt=images[1].shape[:2]
        corners0=np.float64([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        corners1=np.float64([[0,0],[wt-1,0],[wt-1,ht-1],[0,ht-1]])
        # Match the frozen implementation's corner convention exactly.
        c0=mod._image_corners(images[0]);c1=mod._image_corners(images[1])
        low=np.floor(np.vstack([c0,project(c1,h21)]).min(axis=0))
        offset=-low.astype(int)
        x,y=offset
        warped=capture['warped'][1][y:y+h,x:x+w].copy()
        valid=capture['masks'][1][y:y+h,x:x+w]>0
        if warped.shape!=images[0].shape or np.count_nonzero(valid)<64:
            raise ValueError('No useful overlap in reference frame')
        warped[~valid]=0
        record=dict(status='ok',seconds=time.perf_counter()-start,
                    fit_inliers=capture.get('fit_inliers'),h_ref_to_target=h01.tolist(),
                    overlap_fraction=float(valid.mean()))
        if hgt is not None:
            corner_error=np.linalg.norm(project(corners0,h01)-project(corners0,hgt),axis=1)
            record['corner_error_px']=float(np.mean(corner_error))
            prediction=project(grid,h01)
            if 'field' in capture:
                locations=(grid+offset).astype(np.float32)
                delta=cv2.remap(capture['field'],locations[:,0,None],locations[:,1,None],cv2.INTER_LINEAR).reshape(-1,2)
                prediction=project(grid+delta,h01)
            errors=np.linalg.norm(prediction-project(grid,hgt),axis=1)
            record.update(point_count=len(errors),point_median_px=float(np.median(errors)),
                          point_rmse_px=float(np.sqrt(np.mean(errors**2))),
                          point_p95_px=float(np.percentile(errors,95)))
        return record,warped,valid
    except Exception as exc:
        # Explicit identity fallback follows the UDIS literature's failure handling.
        h,w=images[0].shape[:2];ht,wt=images[1].shape[:2]
        warped=np.zeros_like(images[0]);valid=np.zeros((h,w),bool)
        mh,mw=min(h,ht),min(w,wt)
        warped[:mh,:mw]=images[1][:mh,:mw];valid[:mh,:mw]=True
        return dict(status='failed',error=f'{type(exc).__name__}: {exc}',
                    seconds=time.perf_counter()-start,photometric_fallback='identity',
                    corner_error_px=None,point_median_px=None,point_rmse_px=None,point_p95_px=None,
                    overlap_fraction=float(valid.mean())),warped,valid
    finally:
        for name,fn in originals.items():setattr(mod,name,fn)
        cv2.findHomography=original_find
        if original_field:mod._moving_dlt_field=original_field


def evaluate_pair(task):
    if MODULES is None:load_modules()
    paths=[str(ROOT/p) for p in task['images']]
    images=[cv2.imread(p) for p in paths]
    if any(im is None for im in images):raise RuntimeError('Cannot load '+task['id'])
    h,w=images[0].shape[:2];ht,wt=images[1].shape[:2]
    hgt=None;grid=None;gt_mask=None
    if task['dataset']=='hpatches':
        hgt=np.loadtxt(ROOT/task['homography'])
        gx,gy=np.meshgrid(np.linspace(10,w-11,20),np.linspace(10,h-11,20))
        grid=np.column_stack([gx.ravel(),gy.ravel()])
        gt=project(grid,hgt)
        keep=np.isfinite(gt).all(axis=1)&(gt[:,0]>=1)&(gt[:,0]<wt-2)&(gt[:,1]>=1)&(gt[:,1]<ht-2)
        grid=grid[keep]
        gt_mask=cv2.warpPerspective(np.full((ht,wt),255,np.uint8),np.linalg.inv(hgt),(w,h),flags=cv2.INTER_NEAREST)>0
    records=[];warps=[];masks=[]
    for folder,mod in zip(FOLDERS,MODULES):
        record,warped,valid=pair_warp(mod,images,paths,grid,hgt)
        record['method']=folder
        records.append(record);warps.append(warped);masks.append(valid)
    common=gt_mask if hgt is not None else np.logical_and.reduce(masks)
    evaluation_mask=safe_mask(common)
    enough=np.count_nonzero(evaluation_mask)>=256
    for record,warped,valid in zip(records,warps,masks):
        smap=ssim_map(images[0],warped)
        record['psnr_db']=psnr(images[0],warped,common) if enough else None
        record['ssim']=float(smap[evaluation_mask].mean()) if enough else None
        record['scored_area_fraction']=float(common.mean())
        own_safe=safe_mask(valid)
        record['own_overlap_psnr_db']=psnr(images[0],warped,valid)
        record['own_overlap_ssim']=float(smap[own_safe].mean()) if own_safe.any() else None
        if task['dataset']=='udis_d':
            # Reference-style UDIS metric masks both images and averages over the whole frame.
            # Not the main score: padded zeros can inflate these numbers.
            a=images[0].astype(np.float64)*valid[:,:,None]
            b=warped.astype(np.float64)*valid[:,:,None]
            record['masked_canvas_psnr_db']=psnr(a,b,np.ones((h,w),bool))
            record['masked_canvas_ssim']=float(ssim_map(a,b,gaussian=False)[3:-3,3:-3].mean())
    return dict(id=task['id'],dataset=task['dataset'],group=task.get('group'),
                height=h,width=w,common_scoring_valid=enough,methods=records)


def build_manifest():
    tasks=[]
    for seq in sorted((DATA/'hpatches-sequences-release').iterdir()):
        if not seq.is_dir():continue
        for i in range(2,7):
            images=[seq/'1.ppm',seq/f'{i}.ppm']
            gt=seq/f'H_1_{i}'
            tasks.append(dict(dataset='hpatches',id=f'{seq.name}/1-{i}',group=seq.name[0],
                images=[str(p.relative_to(ROOT)) for p in images],
                image_sha256=[sha(p) for p in images],homography=str(gt.relative_to(ROOT)),homography_sha256=sha(gt)))
    for a in sorted((DATA/'testing/input1').glob('*.jpg')):
        b=DATA/'testing/input2'/a.name
        if not b.exists():raise ValueError('Unpaired UDIS image '+a.name)
        tasks.append(dict(dataset='udis_d',id=a.stem,images=[str(p.relative_to(ROOT)) for p in [a,b]],
                          image_sha256=[sha(p) for p in [a,b]]))
    return tasks


def summarize(rows):
    summaries=[]
    for dataset in ['hpatches','udis_d']:
        for folder in FOLDERS:
            subset=[next(m for m in row['methods'] if m['method']==folder) for row in rows if row['dataset']==dataset]
            if not subset:continue
            result=dict(dataset=dataset,method=folder,pairs=len(subset),successful=sum(x['status']=='ok' for x in subset),
                        failed=sum(x['status']!='ok' for x in subset))
            for key in ['psnr_db','ssim','own_overlap_psnr_db','own_overlap_ssim','masked_canvas_psnr_db','masked_canvas_ssim','seconds','scored_area_fraction','overlap_fraction','point_rmse_px']:
                values=[x[key] for x in subset if x.get(key) is not None and math.isfinite(x[key])]
                result['mean_'+key]=float(np.mean(values)) if values else None
                result['n_'+key]=len(values)
            for key in ['corner_error_px','point_median_px','point_p95_px']:
                values=[x[key] for x in subset if x.get(key) is not None and math.isfinite(x[key])]
                result['median_pair_'+key]=float(np.median(values)) if values else None
            if dataset=='hpatches':
                errors=np.array([x.get('corner_error_px') if x.get('corner_error_px') is not None else np.inf for x in subset])
                errors[~np.isfinite(errors)]=np.inf
                for threshold in [1,3,5,10]:
                    result[f'h_success_{threshold}px_pct']=float(np.mean(errors<=threshold)*100)
                    # Exact integral of the empirical recall curve, failures contribute zero.
                    result[f'h_auc_{threshold}px_pct']=float(np.maximum(1-errors/threshold,0).mean()*100)
            summaries.append(result)
    return summaries


def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dataset',choices=['all','hpatches','udis_d'],default='all')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=build_manifest()
    # Results are not shipped with the repository. Generate dataset audit metadata
    # from actual local inputs so the report also works after a fresh clone.
    first={p.name for p in (DATA/'testing/input1').glob('*.jpg')}
    second={p.name for p in (DATA/'testing/input2').glob('*.jpg')}
    sources=json.loads((ROOT/'data/benchmark_sources.json').read_text())
    integrity=dict(
        hpatches_sequences=len({t['id'].split('/')[0] for t in tasks if t['dataset']=='hpatches'}),
        hpatches_pairs=sum(t['dataset']=='hpatches' for t in tasks),
        udis_input1_count=len(first),udis_input2_count=len(second),udis_complete_pairs=len(first&second),
        udis_missing_input1=sorted(second-first),udis_missing_input2=sorted(first-second),
        downloads=sources['downloads'],
        download_hash_note='Expected reference hashes; verify local archives with scripts/prepare_benchmarks.py',
        policy='Pair by identical filenames; never fabricate a missing pair')
    (OUT/'dataset_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    if args.dataset!='all':tasks=[x for x in tasks if x['dataset']==args.dataset]
    if args.limit:tasks=tasks[:args.limit]
    protocol=dict(version=1,scope='Frozen Q1 pairwise geometry/warping benchmark, not multi-image BA or final blend quality',
        evaluator_sha256=sha(Path(__file__)),opencv=cv2.__version__,numpy=np.__version__,workers=args.workers,
        methods={folder:sha(method_code(folder)) for folder in FOLDERS},
        method_paths={folder:str(method_code(folder).relative_to(ROOT)) for folder in FOLDERS},
        selected_counts={name:sum(x['dataset']==name for x in tasks) for name in ['hpatches','udis_d']},
        ground_truth_usage='HPatches GT used only for evaluation, never estimation or APAP controls',
        hpatches_points='20x20 regular reference grid, 10px margins, GT-visible points. Full inverse warp including APAP evaluated in target original pixels.',
        hpatches_corners='Mean forward global homography error at 4 reference corners. APAP uses the preceding global H; therefore same corner error as RootSIFT+MAGSAC++.',
        auc='100 * mean(max(1-error/threshold,0)); exact empirical-CDF integral. Failure error=infinity.',
        hpatches_photometric='Fixed GT overlap; missing estimated coverage is black and penalized. Original uint8 BGR.',
        udis_photometric='Primary: intersection of all five estimated/fallback valid masks in reference frame, before exposure/blending. At least 256 valid SSIM pixels.',
        ssim='11x11 Gaussian, sigma=1.5, K1=.01,K2=.03,data_range=255,population covariance,channel mean; only fully valid windows scored.',
        supplementary_udis='Masked-canvas PSNR and SSIM (7x7 uniform,sample covariance,3px cropped border), approximating author script. Not bit-identical to PyTorch warp.',
        psnr='10log10(255^2/MSE), capped at 100dB for zero MSE',
        failures='Count every failed pair; HPatches accuracy/AUC gets zero credit. Photometric fallback=identity; never silently omit failed pairs.',
        unavailable='UDIS-D has no true global H or independent landmark GT: corner/point accuracy remains null.',
        aggregation='Pair macro means for photometric metrics/RMSE; median of per-pair medians/P95/corner errors for reported median values. HPatches includes 5 pairs per scene.',
        tasks=tasks)
    (OUT/'benchmark_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    path=OUT/'benchmark_pairs.jsonl'
    rows=[]
    if args.resume and path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    done={(x['dataset'],x['id']) for x in rows}
    pending=[x for x in tasks if (x['dataset'],x['id']) not in done]
    start=time.monotonic()
    with path.open('a' if args.resume else 'w',encoding='utf-8') as stream:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=load_modules) as pool:
            for row in pool.map(evaluate_pair,pending,chunksize=1):
                row=clean_json(row)
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                if len(rows)%25==0 or len(rows)==len(tasks):
                    print(f'{len(rows)}/{len(tasks)} pairs, {time.monotonic()-start:.1f}s, last={row["dataset"]}/{row["id"]}',flush=True)
    summary=clean_json(summarize(rows))
    (OUT/'benchmark_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
