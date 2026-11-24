"""Evaluate the 20 SAVED project PNGs in common original-input coordinates.

This is a no-GT source-consistency evaluation, NOT HPatches accuracy or an
official UDIS-D score. Frozen algorithms are replayed only to recover geometry;
every replay must equal the saved PNG pixel for pixel before it is scored.
"""
import argparse
import contextlib
import csv
import hashlib
import html
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from evaluate_benchmarks import psnr, safe_mask, ssim_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/project_evaluation'
METHODS = ['RootSIFT', 'MAGSAC++', 'RootSIFT+MAGSAC++', 'RootSIFT+MAGSAC+++APAP']
MISSIONS = ['Q1', 'Q2', 'Q3', 'E1', 'E2']
MAX_SIDE = 1024


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remap(array, xy, interpolation=cv2.INTER_LINEAR):
    return cv2.remap(array, np.ascontiguousarray(xy[..., 0], dtype=np.float32),
                     np.ascontiguousarray(xy[..., 1], dtype=np.float32), interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def camera_project(xy, k, rotation, scale, projection):
    rays = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
    rays = rays @ (rotation.astype(np.float64) @ np.linalg.inv(k.astype(np.float64))).T
    x, y, z = np.moveaxis(rays, -1, 0)
    if projection == 'plane':
        u, v = scale*x/z, scale*y/z
    elif projection == 'cylindrical':
        u, v = scale*np.arctan2(x, z), scale*y/np.hypot(x, z)
    elif projection == 'spherical':
        u = scale*np.arctan2(x, z)
        v = scale*(np.pi/2 + np.arctan2(y, np.hypot(x, z)))
    else:
        raise ValueError(projection)
    return np.stack([u, v], axis=-1).astype(np.float32)


def homography_project(xy, matrix):
    return cv2.perspectiveTransform(xy.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(xy.shape).astype(np.float32)


def invert_apap(source, field, corner):
    """Solve target + inverse_displacement(target) = source, with residual check."""
    local_source = source - np.asarray(corner, np.float32)
    target = local_source.copy()
    for _ in range(60):
        residual = target + remap(field, target) - local_source
        if float(np.max(np.linalg.norm(residual, axis=-1))) < .02:
            break
        target -= .75 * residual
    residual = np.linalg.norm(target + remap(field, target) - local_source, axis=-1)
    return target + np.asarray(corner, np.float32), residual


def reprojection_origin(valid, origin, scale, kind):
    """Same output origin convention as the frozen cylinder/sphere functions."""
    left, top = origin
    support = valid > 0
    angles = np.arctan((left + np.arange(support.shape[1], dtype=np.float64))/scale)
    occupied = support.any(axis=0)
    upper = top + support.argmax(axis=0)
    lower = top + support.shape[0]-1-support[::-1].argmax(axis=0)
    if kind == 'cylindrical':
        upper, lower = upper*np.cos(angles), lower*np.cos(angles)
    else:
        upper = scale*np.arctan(upper*np.cos(angles)/scale)
        lower = scale*np.arctan(lower*np.cos(angles)/scale)
    return np.array([np.floor((scale*angles)[occupied].min()), np.floor(upper[occupied].min())], np.float32)


def capture_run(mod, paths, label):
    capture = dict(fields=[], matrices=[])
    originals = {}
    def wrap(name, factory):
        originals[name] = getattr(mod, name)
        setattr(mod, name, factory(originals[name]))
    def read_factory(original):
        def read(*args, **kwargs):
            images = original(*args, **kwargs)
            capture['images'] = images
            return images
        return read
    def compose_factory(original):
        def compose(warped, masks, corners, **kwargs):
            result, valid, origin = original(warped, masks, corners, **kwargs)
            capture.update(masks=masks, corners=corners, valid=valid, origin=origin)
            return result, valid, origin
        return compose
    def alignment_factory(original):
        def alignment(features, pairwise, scales, intrinsics, rotations, warper, **kwargs):
            capture.update(intrinsics=intrinsics, rotations=rotations,
                           scale=float(np.median([k[0, 0] for k in intrinsics])), projection='plane')
            return original(features, pairwise, scales, intrinsics, rotations, warper, **kwargs)
        return alignment
    def wide_factory(original):
        def wide(geometry, **kwargs):
            capture.update(intrinsics=geometry.intrinsics, rotations=geometry.rotations,
                           scale=geometry.scale, projection=kwargs['projection'])
            return original(geometry, **kwargs)
        return wide
    def field_factory(original):
        def field(source, target, shape, corner, mask):
            result = original(source, target, shape, corner, mask)
            capture['fields'].append((result[0], corner))
            return result
        return field
    def surface_factory(original, kind):
        def surface(planar, valid, origin, scale, **kwargs):
            capture['reprojection'] = dict(kind=kind, scale=scale,
                origin=reprojection_origin(valid, origin, scale, kind))
            return original(planar, valid, origin, scale, **kwargs)
        return surface
    original_warp = cv2.warpPerspective
    def warp(image, matrix, *args, **kwargs):
        if label == 'Q1' and image.ndim == 3:
            capture['matrices'].append(matrix.copy())
        return original_warp(image, matrix, *args, **kwargs)
    wrap('_read_images', read_factory)
    wrap('_brown_lowe_compose', compose_factory)
    wrap('_log_alignment', alignment_factory)
    wrap('_wide_surface_mosaic', wide_factory)
    wrap('_moving_dlt_field', field_factory)
    wrap('_cylindrical_panorama', lambda f: surface_factory(f, 'cylindrical'))
    wrap('_spherical_panorama', lambda f: surface_factory(f, 'spherical'))
    cv2.warpPerspective = warp
    try:
        replay = getattr(mod, label.lower())(paths)
    finally:
        cv2.warpPerspective = original_warp
        for name, function in originals.items():
            setattr(mod, name, function)
    return replay, capture


def source_samples(saved, capture, label):
    h, w = capture['valid'].shape
    count = np.zeros((h, w), np.uint8)
    origin = np.asarray(capture['origin'])
    for mask, corner in zip(capture['masks'], capture['corners']):
        x, y = (np.asarray(corner)-origin).astype(int)
        mh, mw = mask.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x+mw), min(h, y+mh)
        if x1>x0 and y1>y0:
            count[y0:y1, x0:x1] += (mask[y0-y:y1-y, x0-x:x1-x]>0).astype(np.uint8)
    # Require a full neighbourhood in overlap, avoiding black padding and warp borders.
    overlap = safe_mask(count >= 2, 5).astype(np.uint8)
    samples = []
    fields = capture['fields']
    for i, image in enumerate(capture['images']):
        ih, iw = image.shape[:2]
        factor = min(1., MAX_SIDE/max(ih, iw))
        gh, gw = max(11, round(ih*factor)), max(11, round(iw*factor))
        xx, yy = np.meshgrid(np.linspace(0, iw-1, gw, dtype=np.float32),
                             np.linspace(0, ih-1, gh, dtype=np.float32))
        xy = np.stack([xx, yy], axis=-1)
        ref = remap(image, xy)
        if label == 'Q1':
            assert len(capture['matrices']) == 2
            global_xy = homography_project(xy, capture['matrices'][i])
        else:
            global_xy = camera_project(xy, capture['intrinsics'][i], capture['rotations'][i],
                                       capture['scale'], capture['projection'])
            # Validate the vectorized formula against the actual OpenCV forward projector.
            warper = cv2.PyRotationWarper(capture['projection'], capture['scale'])
            for row, col in [(gh//2, gw//2), (gh//4, gw//4), (3*gh//4, 3*gw//4)]:
                expected = warper.warpPoint(tuple(map(float, xy[row, col])),
                    capture['intrinsics'][i], capture['rotations'][i])
                assert np.linalg.norm(global_xy[row, col]-expected)<.02, 'Camera mapping mismatch'
        residual = np.zeros((gh, gw), np.float32)
        if fields and (label != 'Q1' or i == 1):
            field, corner = fields[0 if label == 'Q1' else i]
            global_xy, residual = invert_apap(global_xy, field, corner)
        local_xy = global_xy-origin
        eligible = remap(overlap, local_xy, cv2.INTER_NEAREST)>0
        # The particular source must itself still cover this position after APAP.
        own = safe_mask(capture['masks'][i]>0, 5).astype(np.uint8)
        eligible &= remap(own, global_xy-np.asarray(capture['corners'][i]), cv2.INTER_NEAREST)>0
        eligible &= residual < .1
        if 'reprojection' in capture:
            rp = capture['reprojection'];f = rp['scale']
            theta = np.arctan(global_xy[..., 0]/f)
            v = global_xy[..., 1]*np.cos(theta)
            if rp['kind'] == 'spherical':
                v = f*np.arctan(v/f)
            final_xy = np.stack([f*theta, v], axis=-1)-rp['origin']
        else:
            final_xy = local_xy
        eligible &= np.isfinite(final_xy).all(axis=-1)
        eligible &= (final_xy[..., 0]>=1)&(final_xy[..., 0]<saved.shape[1]-2)
        eligible &= (final_xy[..., 1]>=1)&(final_xy[..., 1]<saved.shape[0]-2)
        samples.append(dict(reference=ref, reconstructed=remap(saved, final_xy), valid=eligible,
                            apap_inverse_rejected=int(np.count_nonzero(residual>=.1)),
                            original_shape=[ih, iw], sample_shape=[gh, gw]))
    return samples


def evaluate_one(folder, label):
    base = ROOT/'results'/folder
    manifest = json.loads((base/'experiment.json').read_text())
    record = next(r for r in manifest['missions'] if r['output'] == label+'_result.png')
    saved_path = base/record['output']
    assert sha(saved_path) == record['output_sha256'], f'PNG changed: {saved_path}'
    assert sha(base/'stitching_snapshot.py') == manifest['snapshot_sha256']
    paths = [str(ROOT/item['path']) for item in record['inputs']]
    for item in record['inputs']:
        assert sha(ROOT/item['path']) == item['sha256'], f'Input changed: {item["path"]}'
    spec = importlib.util.spec_from_file_location('frozen_project_method', base/'stitching_snapshot.py')
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.VERBOSE = False
    cv2.setNumThreads(manifest['opencv_threads'])
    with (OUT/'logs'/f'{folder}_{label}.log').open('w') as log, contextlib.redirect_stdout(log):
        replay, capture = capture_run(mod, paths, label)
    saved = cv2.imread(str(saved_path))
    assert np.array_equal(replay, saved), f'Replay differs from saved PNG: {folder} {label}'
    samples = source_samples(saved, capture, label)
    metadata = dict(method=folder, mission=label, png=str(saved_path.relative_to(ROOT)),
        png_sha256=sha(saved_path), snapshot_sha256=sha(base/'stitching_snapshot.py'),
        inputs=record['inputs'], replay_pixel_exact=True, width=saved.shape[1], height=saved.shape[0],
        independent_point_error_px=None, corner_error_px=None, h_auc_5px=None,
        geometry_accuracy_unavailable='No independent geometric ground truth for these project photos',
        postprocessing=record['postprocessing'])
    return metadata, samples


def export_csv(path, records):
    keys = list(dict.fromkeys(k for record in records for k in record))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader();writer.writerows(records)


def diagnostic_image(path, sample, mask):
    ref, restored = sample['reference'], sample['reconstructed']
    difference = np.abs(ref.astype(np.float32)-restored.astype(np.float32)).mean(axis=2)
    heat = cv2.applyColorMap(np.uint8(np.clip(difference/50, 0, 1)*255), cv2.COLORMAP_TURBO)
    panels = []
    for image, title in [(ref, 'Original input'), (restored, 'SAVED PNG mapped to input'),
                         (heat, 'Mean absolute error: 0..50+ / 255')]:
        im = image.copy();im[~mask] = 35
        scale = min(1., 420/im.shape[1])
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        im = cv2.copyMakeBorder(im, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245,245,245))
        cv2.putText(im, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .46, (25,25,25), 1, cv2.LINE_AA)
        panels.append(im)
    assert cv2.imwrite(str(path), np.hstack(panels))


def report(records):
    export_csv(OUT/'summary.csv', [{k:v for k,v in r.items() if not isinstance(v, (list, dict))} for r in records])
    export_csv(OUT/'per_source.csv', [dict(method=r['method'], mission=r['mission'], **s)
                                    for r in records for s in r['sources']])
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>프로젝트 최종 PNG 20개 평가</title>',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<style>body{font:16px/1.7 system-ui;color:#172536;max-width:1400px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border:1px solid #cbd5e1;text-align:left}th{background:#eef3f8}img{max-width:100%}.note{background:#eef5ff;padding:16px}a{color:#075cab}</style>',
        '<h1>프로젝트 최종 PNG 20개 평가</h1>',
        '<p class="note">평가 대상은 네 방법 × Q1/Q2/Q3/E1/E2의 저장된 최종 PNG 20개다. 외부 HPatches/UDIS-D 영상은 사용하지 않았다. 원본 입력과의 일치도를 평가하며, 정답 파노라마 대비 정확도나 공식 벤치마크 점수가 아니다.</p>',
        '<p><a href="summary.csv">20개 점수 CSV</a> · <a href="per_source.csv">입력별 상세 CSV</a> · <a href="protocol.json">프로토콜</a> · <a href="results.json">결과·해시</a> · <a href="../comparison.html">원본 결과 이미지 비교</a></p>',
        '<h2>평가 방식</h2><ol><li>저장된 코드·원본 사진으로 정합 변환을 복원하고, 재실행 결과가 저장된 PNG와 픽셀 단위로 동일한지 검사한다.</li>',
        '<li>각 원본의 격자점을 실제 카메라/호모그래피/APAP 변환과 최종 원통·구면 투영을 통해 PNG 위치로 옮겨, <strong>디스크에서 읽은 최종 PNG</strong>의 색상을 샘플링한다.</li>',
        '<li>네 방법 모두 두 장 이상 중첩되는 원본 좌표를 공통 평가 영역으로 고정한다. E1/E2는 여덟 결과 모두 같은 영역을 사용한다. 검정 여백과 경계는 제외한다.</li>',
        '<li>원본과 복원 영상의 PSNR/SSIM을 입력별로 계산하고 동일 가중치로 평균한다. 최대 변 길이 1,024의 동일 좌표 격자를 사용하며, 별도의 노출 정규화나 추가 정합은 하지 않는다.</li></ol>',
        '<p>APAP의 역변위 필드는 수치적으로 뒤집고 잔차가 0.1 출력 픽셀 미만인 점만 사용한다. 이것은 좌표 계산 검증 기준이며 정합의 실제 위치 오차가 아니다. 원본→파노라마→원본의 보간 손실도 점수에 포함된다.</p>',
        '<h2>20개 최종 결과의 원본 일치도</h2>',
        '<p>PSNR(dB)·SSIM은 높을수록 원본과 비슷하다. 원본 좌표의 공통 겹침 영역에서 평가했다. <strong>원본을 그대로 선택한 영역은 점수가 높아질 수 있고, 블렌딩의 평균화도 점수를 높일 수 있다. 반대로 E2의 의도적인 노출 보정은 점수를 낮출 수 있다.</strong> 그러므로 자연스러움·이음선 제거·고스팅 개선의 종합 순위로 해석하지 않는다.</p>',
        table(['미션','방법','최종 PNG↔원본 PSNR ↑','최종 PNG↔원본 SSIM ↑','입력별 공통 평가 면적 평균(%)'],
              [[r['mission'],r['method'],f"{r['source_consistency_psnr_db']:.3f}",f"{r['source_consistency_ssim']:.4f}",f"{r['common_area_fraction']*100:.2f}"] for r in records]),
        '<h2>E1 → E2: 동일 영역에서의 후처리 영향</h2>',
        table(['방법','PSNR 변화(dB)','SSIM 변화'], [[folder,
            f"{next(r for r in records if r['method']==folder and r['mission']=='E2')['source_consistency_psnr_db']-next(r for r in records if r['method']==folder and r['mission']=='E1')['source_consistency_psnr_db']:+.3f}",
            f"{next(r for r in records if r['method']==folder and r['mission']=='E2')['source_consistency_ssim']-next(r for r in records if r['method']==folder and r['mission']=='E1')['source_consistency_ssim']:+.4f}"] for folder in METHODS]),
        '<h2>정답이 없어 산출하지 않은 항목</h2><p>독립 대응점 위치 오차, Corner error, H 성공률/AUC는 산출 불가(null)다. 정합 코드가 스스로 추정한 변환을 정답으로 사용하지 않았다. 모든 원본 위치가 서로 어떻게 대응하는지에 대한 독립 정답이 없기 때문이다. 성공적인 PNG 재현 20/20은 정합 정확도 100%를 의미하지 않는다.</p>',
        '<p>기존 외부 데이터셋 점수와 이 점수는 데이터·참조·평가 영역이 다르므로 직접 비교하지 않는다. 미션 간 총점이나 순위도 만들지 않았다. 이전 mission_summary.csv는 최종 PNG가 아닌 후처리 전 입력 간 일치도이며 이 표와 다른 지표다.</p>',
        '<h2>대표 입력의 비교와 오차 지도</h2><p>같은 미션의 네 방법은 같은 원본 사진·같은 평가 마스크를 보여 준다. 회색은 제외 영역이다. 오차 지도는 채널 평균 절대 차이 0~50 이상/255를 공통 색상 범위로 표시한다.</p>']
    for r in records:
        preview = r['png'].replace(f'/{r["mission"]}_result.png', f'/previews/{r["mission"]}.jpg')
        parts.append(f'<h3>{html.escape(r["method"])} / {r["mission"]} / 입력 {r["diagnostic_source"]}</h3><a href="../../{html.escape(preview)}">평가한 결과의 미리보기</a><br><img loading="lazy" src="{html.escape(r["diagnostic_image"])}">')
    parts.append('<p>재현: <code>python3 -B experiments/evaluate_project_results.py</code>. 기존 생성 환경의 BLAS 스레드 설정을 유지해야 픽셀 단위 재현 검사를 통과한다.</p></html>')
    (OUT/'report.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Check saved-PNG geometry for one method on all missions')
    args = parser.parse_args()
    for sub in ['logs', 'diagnostics', 'masks']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    if args.pilot:
        for label in MISSIONS:
            metadata, samples = evaluate_one(METHODS[-1], label)
            print(label, 'pixel-exact', metadata['replay_pixel_exact'], 'overlap fractions',
                  [round(float(s['valid'].mean()), 3) for s in samples], flush=True)
        return
    preserved = {str(p.relative_to(ROOT)):sha(p) for folder in [*METHODS, 'brown+lowe']
                 for p in (ROOT/'results'/folder).iterdir() if p.suffix=='.png' or p.name=='stitching_snapshot.py'}
    records = []
    # E1 and E2 intentionally share the exact same original-input scoring masks.
    for labels in [['Q1'], ['Q2'], ['Q3'], ['E1', 'E2']]:
        runs = []
        for label in labels:
            for folder in METHODS:
                metadata, samples = evaluate_one(folder, label)
                runs.append((metadata, samples))
                print('Recovered SAVED PNG:', folder, label, 'pixel-exact', flush=True)
        n = len(runs[0][1])
        common = [np.logical_and.reduce([samples[i]['valid'] for _, samples in runs]) for i in range(n)]
        representative = int(np.argmax([mask.mean() for mask in common]))
        for i, mask in enumerate(common):
            assert np.count_nonzero(safe_mask(mask))>=256, 'Insufficient common overlap'
            assert cv2.imwrite(str(OUT/'masks'/f'{"_".join(labels)}_input{i+1}.png'), np.uint8(mask)*255)
        for metadata, samples in runs:
            source_rows = []
            for i, (sample, mask) in enumerate(zip(samples, common)):
                ref, restored = sample['reference'], sample['reconstructed']
                # All method comparisons must have precisely identical original references.
                assert np.array_equal(ref, runs[0][1][i]['reference'])
                safe = safe_mask(mask)
                source_rows.append(dict(source=i+1, input=metadata['inputs'][i]['path'],
                    psnr_db=psnr(ref, restored, mask), ssim=float(ssim_map(ref, restored)[safe].mean()),
                    common_pixels=int(mask.sum()), ssim_pixels=int(safe.sum()),
                    common_area_fraction=float(mask.mean()),
                    evaluation_width=ref.shape[1], evaluation_height=ref.shape[0],
                    apap_inverse_rejected=sample['apap_inverse_rejected']))
            metadata.update(sources=source_rows,
                source_consistency_psnr_db=float(np.mean([r['psnr_db'] for r in source_rows])),
                source_consistency_ssim=float(np.mean([r['ssim'] for r in source_rows])),
                common_area_fraction=float(np.mean([r['common_area_fraction'] for r in source_rows])),
                diagnostic_source=representative+1,
                diagnostic_image=f'diagnostics/{metadata["method"]}_{metadata["mission"]}.png')
            diagnostic_image(OUT/metadata['diagnostic_image'], samples[representative], common[representative])
            records.append(metadata)
            print('SCORED', metadata['method'], metadata['mission'],
                  round(metadata['source_consistency_psnr_db'], 3), round(metadata['source_consistency_ssim'], 4), flush=True)
        del runs
        (OUT/'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    assert len(records)==20
    assert all(sha(ROOT/p)==digest for p,digest in preserved.items()), 'Existing result changed'
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluator_sha256=sha(Path(__file__)),
        opencv=cv2.__version__, numpy=np.__version__, evaluated_pngs=20, methods=METHODS,
        scope='SAVED final PNG source consistency in common original-input overlap coordinates; no external dataset',
        independent_geometric_ground_truth=False, comparison_reference='Original uncorrected input pixels',
        sample_grid='Endpoint-aligned original input coordinates; longest side <=1024; bilinear sampling',
        geometry='Frozen exact replay -> H/camera projection -> APAP inverse-field solve -> final cylinder/sphere -> crop',
        apap_inverse='Damped fixed-point, up to 60 iterations; mapping residual <0.1px; residual NOT accuracy',
        mask='>=2 contributing sources; 5x5 eroded overlap and source support; intersection of four methods, or all eight E1/E2 results',
        psnr='255 range; 100dB cap at zero MSE; all common pixels; macro average over original inputs',
        ssim='11x11 Gaussian sigma1.5, K1=.01 K2=.03, population covariance, BGR channel average; full valid windows only',
        aggregation='Equal weight per input image within a mission; no cross-mission aggregate',
        caveats=['Not HPatches/UDIS-D official score; not geometric truth or aesthetic quality',
                 'Selection/copying and averaging can increase consistency; intended gain changes can decrease it',
                 'Round-trip interpolation loss is included; no extra alignment or exposure normalization',
                 'Common scoring domain depends on the compared methods'],
        preserved_files=preserved)
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    report(records)
    print('Wrote', OUT/'report.html', flush=True)


if __name__ == '__main__':
    main()
