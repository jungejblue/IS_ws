"""Photometric diagnostics for the 25 frozen mission runs; no landmark GT invented."""
import json
from pathlib import Path
import cv2
import numpy as np

import evaluate_benchmarks as evaluation


def layers_for(mod,paths,label):
    original=mod._brown_lowe_compose
    captured={}
    def capture(warped,masks,corners,**kwargs):
        captured.update(warped=warped,masks=masks,corners=corners)
        return original(warped,masks,corners,**kwargs)
    mod._brown_lowe_compose=capture
    try:
        getattr(mod,label.lower())(paths)
    finally:
        mod._brown_lowe_compose=original
    return captured


def pair_metric(layers,i,j):
    images,masks,corners=layers['warped'],layers['masks'],layers['corners']
    ci,cj=np.array(corners[i]),np.array(corners[j])
    si=np.array(images[i].shape[1::-1]);sj=np.array(images[j].shape[1::-1])
    low=np.maximum(ci,cj);high=np.minimum(ci+si,cj+sj)
    if np.any(high-low<11):return None
    x,y=low-ci;w,h=high-low
    a=images[i][y:y+h,x:x+w];ma=masks[i][y:y+h,x:x+w]>0
    x,y=low-cj
    b=images[j][y:y+h,x:x+w];mb=masks[j][y:y+h,x:x+w]>0
    valid=ma&mb;safe=evaluation.safe_mask(valid)
    if np.count_nonzero(safe)<256:return None
    return dict(pair=f'{i+1}-{j+1}',pixels=int(valid.sum()),
                psnr_db=evaluation.psnr(a,b,valid),ssim=float(evaluation.ssim_map(a,b)[safe].mean()))


def main():
    evaluation.load_modules()
    root=evaluation.ROOT
    records=[]
    for folder,mod in zip(evaluation.FOLDERS,evaluation.MODULES):
        for label in ['Q1','Q2','Q3','E1','E2']:
            data=root/'data'/('Q3' if label.startswith('E') else label)
            paths=sorted(str(p) for p in data.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'])
            if hasattr(mod,'EXPERIMENT_DIAGNOSTICS'):mod.EXPERIMENT_DIAGNOSTICS.clear()
            layers=layers_for(mod,paths,label)
            n=len(paths)
            pairs=[(i,j) for i in range(n) for j in range(i+1,n)
                   if n<8 or max(abs(i//4-j//4),abs(i%4-j%4))<=1]
            metrics=[]
            for i,j in pairs:
                value=pair_metric(layers,i,j)
                if value is not None:metrics.append(value)
            records.append(dict(method=folder,mission=label,pairs=metrics,
                independent_point_error=None,corner_error=None,
                unavailable_reason='No independent landmark or ground-truth homography for project photos',
                stage='After geometric warp/APAP, before gain/seam/blending',
                note='Method-specific overlap in its own projection; diagnostic, not a fixed-GT benchmark'))
            del layers
            print(folder,label,'overlaps',len(metrics),flush=True)
    for label in ['Q1','Q2','Q3','E1','E2']:
        subset=[r for r in records if r['mission']==label]
        common=set.intersection(*[{p['pair'] for p in r['pairs']} for r in subset])
        for r in subset:
            selected=[p for p in r['pairs'] if p['pair'] in common]
            r['common_pair_ids']=sorted(common)
            r['mean_overlap_psnr_db']=float(np.mean([p['psnr_db'] for p in selected])) if selected else None
            r['mean_overlap_ssim']=float(np.mean([p['ssim'] for p in selected])) if selected else None
    path=evaluation.OUT/'mission_photometric.json'
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print('Saved',path,flush=True)


if __name__=='__main__':main()
