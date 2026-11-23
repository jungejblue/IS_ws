"""이미지 스티칭 과제: Q1 / Q2 / Q3 / E1 / E2.

Q1: SIFT -> ratio test -> RANSAC homography -> planar warp -> seam -> feather.
Q2: SIFT -> pairwise matching -> camera estimation -> ray bundle adjustment
    -> planar warp -> optional exposure compensation / seam / multi-band blending
    -> cylindrical reprojection (no crop).

Brown & Lowe, Automatic Panoramic Image Stitching using Invariant Features,
IJCV, 2007: https://www.cs.ubc.ca/~lowe/papers/07brown.pdf
OpenCV 공식 단계별 구성 참고:
https://github.com/opencv/opencv/blob/4.x/samples/python/stitching_detailed.py

논문 전체의 재현은 아닙니다. Q1은 graph-cut seam과 feather blending을 사용하고,
Q2/Q3의 블렌딩 옵션은 노출 보정, 이음선 선택, multi-band 합성을 함께 제어합니다.
현재 기본값은 꺼져 있으며 원래 유효 마스크로 단순 덮어쓰기합니다. Stitcher API는 사용하지 않습니다.
모든 입력은 img_paths에서만 읽으며, 파일 저장 및 GUI 호출은 하지 않습니다.
Q3는 2x4 격자의 평면 정합 후 전체를 원통 재투영합니다.
단일 평면의 투영 범위를 넘는 Q3 입력은 모든 영상을 공통 원통에서 합성합니다.
평행이동으로 설명되는 입력은 이동 모델, 나머지는 카메라 회전 모델을 사용합니다.
E1은 같은 8장의 평면 정합 + feather 합성 후 구면 재투영을 수행합니다.
E2는 블록 노출 보정 + multi-band 합성 후 동일한 구면 좌표계로 재투영합니다.
E1/E2도 단일 평면의 범위를 넘으면 전체 입력을 공통 구면 좌표계에서 합성합니다.
"""

import cv2
import numpy as np


# 알고리즘 설정: 비교 실험에서는 동일한 값으로 고정하세요.
WORK_MEGAPIX = 1.0        # 특징점 처리용 영상의 최대 면적; 출력은 원본 축척
SIFT_NFEATURES = 6000
Q1_RATIO = 0.75
Q1_RANSAC_PX = 3.0       # 특징점을 계산한 축소 영상 좌표 기준
MIN_INLIERS = 10
Q2_MATCH_CONF = 0.65     # OpenCV detail matcher 설정; Q1_RATIO와 의미가 다름
Q2_GRAPH_CONF = 1.0      # 연결 그래프 및 bundle adjustment의 confidence 기준
Q2_WORK_MEGAPIX = 1.0   # Q1과 독립적으로 특징점 처리 해상도를 실험
Q2_SEAM_COST = "COST_COLOR_GRAD"
Q2_SEAM_DILATE_RADIUS = 3
Q2_EXPOSURE_BLOCK = 128 # 원본 축척의 평면 워핑 영상 기준; 작은 블록의 과보정 억제
Q2_BLEND_BANDS = 5      # 넓은 밝기 변화와 세부 윤곽을 다른 해상도에서 합성
Q2_USE_BLENDING = False # False: 노출 보정/이음선/블렌딩 모두 끄고 단순 덮어쓰기
Q3_USE_BLENDING = False # True: 노출 보정 + 이음선 선택 + multi-band 합성
Q3_WORK_MEGAPIX = 1.0
Q3_MATCH_CONF = 0.45
Q3_GRAPH_CONF = 1.0
Q3_RATIO = 0.75
Q3_TRANSLATION_P95 = 0.75  # 작업 영상 px; 단순 이동 모델을 채택하는 오차 한계
Q3_FOCAL_WIDTH_FACTOR = 1.0  # 이동만 있는 입력은 focal 식별 불가: 출력 너비 기준 표시 축척
Q3_WIDE_SEAM_MEGAPIX = 0.12  # 넓은 원통 합성의 이음선 탐색만 축소; 출력은 원본 축척
FEATHER_WIDTH = 32.0     # 최종 출력 좌표에서 경계 가중치를 증가시키는 폭(px)
USE_GRAPH_CUT = True     # 이음선을 선택하여 시차 영역의 이중 윤곽을 줄임
SEAM_DILATE_RADIUS = 3   # 선택된 이음선 주변에만 작은 블렌딩 영역을 확보
MAX_CANVAS_PIXELS = 24_000_000
MAX_SIDE = 30_000        # OpenCV remap의 크기 제한 및 과도한 메모리 사용 방지
VERBOSE = True
# 실험 실행기가 선택하는 과제용 Brown–Lowe 기반 프로필.
# 기본 실행은 기존 설정을 유지한다. 원 논문 전체를 재현한 구현은 아니다.
BASELINE_PROFILE = None
ALGORITHM = "brown_lowe"  # rootsift / magsac / rootsift_magsac / rootsift_magsac_apap
EXPERIMENT_DIAGNOSTICS = []
APAP_GRID_STEP = 64
APAP_SIGMA = 180.0       # 원본 축척의 공통 투영 좌표에서 거리(px)
APAP_WEIGHT_FLOOR = 0.01
APAP_MAX_CORRECTION = 48.0


def _log(message):
    if VERBOSE:
        print(message)


class _PlanarProjectionError(ValueError):
    """평면 지평선을 넘은 경우 이미 추정한 카메라를 원통/구면 합성에서 재사용."""
    def __init__(self, message, images, intrinsics, rotations, scale, alignment=None):
        super().__init__(message)
        self.images = images
        self.intrinsics = intrinsics
        self.rotations = rotations
        self.scale = scale
        self.alignment = alignment


def _root_descriptors(descriptors):
    if "rootsift" not in ALGORITHM:
        return descriptors
    values = descriptors.astype(np.float32)
    return np.sqrt(values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12))


def _homography_method():
    return cv2.USAC_MAGSAC if "magsac" in ALGORITHM else cv2.RANSAC


class _MatchRecord:
    """Python 바인딩에서 수정할 수 없는 MatchesInfo의 inlier 목록을 별도 보관."""
    def __init__(self, native, matches, inliers):
        self.src_img_idx, self.dst_img_idx = native.src_img_idx, native.dst_img_idx
        self.H, self.confidence = native.H, float(native.confidence)
        self.num_inliers = int(np.count_nonzero(inliers))
        self.matches, self.inliers = matches, np.asarray(inliers, np.uint8).ravel()

    def getMatches(self):
        return self.matches

    def getInliers(self):
        return self.inliers


def _experiment_matches(features, native, *, label):
    """전체 매칭 후보에 MAGSAC++를 적용하고 H와 inlier를 후속 단계에 전달."""
    records = []
    count = len(features)
    for match in native:
        i, j = int(match.src_img_idx), int(match.dst_img_idx)
        if not 0 <= i < j < count:
            continue
        matches = match.getMatches()
        keep = np.asarray(match.getInliers(), np.uint8)
        if "magsac" in ALGORITHM and len(matches) >= 4:
            pi = np.float64([features[i].getKeypoints()[m.queryIdx].pt for m in matches])
            pj = np.float64([features[j].getKeypoints()[m.trainIdx].pt for m in matches])
            # OpenCV detail estimator는 영상 중심 기준 H를 사용한다.
            pi -= np.asarray(features[i].img_size) / 2
            pj -= np.asarray(features[j].img_size) / 2
            h, mask = cv2.findHomography(pi, pj, cv2.USAC_MAGSAC, 3.0,
                                         maxIters=5000, confidence=0.995)
            keep = np.zeros(len(matches), np.uint8) if mask is None else mask.ravel()
            match.num_inliers = int(keep.sum())
            match.confidence = match.num_inliers / (8 + 0.3 * len(matches))
            if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
                match.confidence = 0.0
                keep[:] = 0
                match.num_inliers = 0
                native[j * count + i].confidence = 0.0
                native[j * count + i].num_inliers = 0
            else:
                match.H = h
                reverse = native[j * count + i]
                reverse.H = np.linalg.inv(h)
                reverse.num_inliers, reverse.confidence = match.num_inliers, match.confidence
        if len(keep) != len(matches):
            keep = np.zeros(len(matches), np.uint8)
        records.append(_MatchRecord(match, matches, keep))
    name = "MAGSAC++" if "magsac" in ALGORITHM else "RANSAC"
    _log(f"[{label}] algorithm={ALGORITHM}, robust estimator={name}; selected inliers passed to ray BA")
    EXPERIMENT_DIAGNOSTICS.append({"stage": "matching", "algorithm": ALGORITHM,
        "estimator": name, "pairs": [{"i": r.src_img_idx, "j": r.dst_img_idx,
        "candidates": len(r.matches), "inliers": r.num_inliers,
        "confidence": r.confidence} for r in records]})
    return records


def _ray_bundle_adjust(features, pairwise, cameras, graph_conf, *, label):
    """NumPy LM으로 OpenCV와 같은 종류의 focal-scaled ray residual 최소화.

    네 실험에 동일한 solver를 사용한다. native MatchesInfo의 read-only inlier
    제약을 우회하기 위해 구현했으며 원 논문/OpenCV optimizer의 완전 재현은 아니다.
    """
    n = len(cameras)
    edges = []
    for pair in pairwise:
        if pair.confidence <= graph_conf or pair.num_inliers < MIN_INLIERS:
            continue
        i, j = pair.src_img_idx, pair.dst_img_idx
        matches = [m for m, keep in zip(pair.matches, pair.inliers) if keep]
        p = np.float64([features[i].getKeypoints()[m.queryIdx].pt for m in matches])
        q = np.float64([features[j].getKeypoints()[m.trainIdx].pt for m in matches])
        p -= [cameras[i].ppx, cameras[i].ppy]
        q -= [cameras[j].ppx, cameras[j].ppy]
        edges.append((i, j, p, q))
    if not edges:
        raise ValueError(f"{label}: no edges for ray bundle adjustment")
    rotations0 = [cv2.Rodrigues(c.R.astype(np.float64))[0].ravel() for c in cameras]
    # 카메라 0의 회전 고정으로 전체 회전 gauge freedom 제거.
    params = np.r_[np.log([c.focal for c in cameras]), np.concatenate(rotations0[1:])]
    def unpack(x):
        focal = np.exp(x[:n])
        rotations = [cv2.Rodrigues(rotations0[0])[0]]
        rotations += [cv2.Rodrigues(x[n+3*(k-1):n+3*k])[0] for k in range(1, n)]
        return focal, rotations
    def residual(x):
        focal, rotations = unpack(x)
        chunks = []
        for i, j, p, q in edges:
            a = np.column_stack((p, np.full(len(p), focal[i]))) @ rotations[i].T
            b = np.column_stack((q, np.full(len(q), focal[j]))) @ rotations[j].T
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            chunks.append((np.sqrt(focal[i] * focal[j]) * (a-b)).ravel())
        return np.concatenate(chunks)
    r = residual(params)
    initial_cost = cost = float(r @ r)
    damping = 1e-3
    accepted = 0
    for iteration in range(70):
        jac = np.empty((r.size, params.size))
        for k in range(params.size):
            delta = 1e-6 * max(1.0, abs(params[k]))
            trial = params.copy()
            trial[k] += delta
            jac[:, k] = (residual(trial) - r) / delta
        normal, grad = jac.T @ jac, jac.T @ r
        diag = np.maximum(np.diag(normal), 1e-8)
        previous_cost = cost
        updated = False
        for _ in range(12):
            step = np.linalg.solve(normal + np.diag(damping * diag), -grad)
            trial = params + step
            # 비물리적인 수치 발산을 거부하며 사진별 초점거리 수동 튜닝은 하지 않는다.
            if np.any(trial[:n] < np.log(10)) or np.any(trial[:n] > np.log(1e5)):
                damping *= 10
                continue
            candidate = residual(trial)
            candidate_cost = float(candidate @ candidate)
            if np.isfinite(candidate_cost) and candidate_cost < cost:
                params, r, cost = trial, candidate, candidate_cost
                damping = max(damping / 3, 1e-10)
                accepted += 1
                updated = True
                break
            damping *= 10
        if not updated or previous_cost-cost < 1e-8 * max(previous_cost, 1.0):
            break
    focal, rotations = unpack(params)
    for camera, f, rotation in zip(cameras, focal, rotations):
        camera.focal = float(f)
        camera.R = rotation.astype(np.float32)
    before, after = np.sqrt(initial_cost / (r.size / 3)), np.sqrt(cost / (r.size / 3))
    _log(f"[{label}] NumPy ray BA: fitting RMS {before:.4f} -> {after:.4f}px, steps={accepted}")
    EXPERIMENT_DIAGNOSTICS.append({"stage": "ray_ba", "solver": "numpy_lm",
        "initial_fitting_rms": float(before), "final_fitting_rms": float(after),
        "accepted_steps": accepted, "note": "training residual, not held-out accuracy"})
    return cameras


def _apap_tracks(features, pairwise, scales, intrinsics, rotations, warper, graph_conf):
    """동일 특징점 track의 평균 투영 위치를 모든 영상의 공통 목표로 사용."""
    parent = {}
    def root(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = root(parent[node])
        return parent[node]
    for pair in pairwise:
        if pair.confidence <= graph_conf or pair.num_inliers < MIN_INLIERS:
            continue
        for match, keep in zip(pair.getMatches(), pair.getInliers()):
            if keep:
                a, b = (pair.src_img_idx, match.queryIdx), (pair.dst_img_idx, match.trainIdx)
                parent[root(a)] = root(b)
    groups = {}
    for node in parent:
        groups.setdefault(root(node), []).append(node)
    points = [[] for _ in features]
    targets = [[] for _ in features]
    for nodes in groups.values():
        # 같은 영상의 서로 다른 점이 포함된 불일치 track은 사용하지 않는다.
        if len(nodes) < 2 or len({i for i, _ in nodes}) != len(nodes):
            continue
        projected = []
        for i, k in nodes:
            p = np.asarray(features[i].getKeypoints()[k].pt) / np.diag(scales[i])[:2]
            projected.append(warper.warpPoint(tuple(map(float, p)), intrinsics[i], rotations[i]))
        projected = np.asarray(projected)
        target = projected.mean(axis=0)
        if not np.isfinite(projected).all() or np.max(np.linalg.norm(projected-target, axis=1)) > APAP_MAX_CORRECTION:
            continue
        for (i, _), p in zip(nodes, projected):
            points[i].append(p)
            targets[i].append(target)
    return [(np.asarray(p, np.float64).reshape(-1, 2), np.asarray(q, np.float64).reshape(-1, 2))
            for p, q in zip(points, targets)]


def _moving_dlt_field(source, target, shape, corner, mask):
    """Inverse Moving DLT: shared target -> globally warped source.

    격자마다 정규화 DLT를 푼다. 항등변환 앵커, 지지영역 밖 감쇠, 변위 상한과
    접힘 방지를 추가한 APAP 계열 구현이며 원 논문의 BAMDLT 재현은 아니다.
    """
    h, w = shape
    scale = float(max(w, h))
    origin = np.asarray(corner, np.float64)
    src = source - origin
    dst = target - origin
    # 대응점이 없는 곳에는 항등변환 제약을 약하게 적용한다.
    anchors = np.float64([[0,0],[w-1,0],[0,h-1],[w-1,h-1],
                         [(w-1)/2,0],[(w-1)/2,h-1],[0,(h-1)/2],[w-1,(h-1)/2]])
    center = np.float64([(w-1)/2, (h-1)/2])
    p = (np.vstack([dst, anchors])-center) / scale
    q = (np.vstack([src, anchors])-center) / scale
    x, y, u, v = p[:,0], p[:,1], q[:,0], q[:,1]
    zeros, ones = np.zeros(len(p)), np.ones(len(p))
    ax = np.column_stack([-x,-y,-ones,zeros,zeros,zeros,u*x,u*y,u])
    ay = np.column_stack([zeros,zeros,zeros,-x,-y,-ones,v*x,v*y,v])
    design = np.stack([ax,ay],axis=1).reshape(-1,9)
    nx, ny = max(2,int(np.ceil((w-1)/APAP_GRID_STEP))+1), max(2,int(np.ceil((h-1)/APAP_GRID_STEP))+1)
    gx, gy = np.meshgrid(np.linspace(0,w-1,nx), np.linspace(0,h-1,ny))
    nodes = np.column_stack([gx.ravel(),gy.ravel()])
    offsets = np.zeros_like(nodes)
    max_raw = 0.0
    for k, node in enumerate(nodes):
        d2 = np.sum((dst-node)**2, axis=1)
        weights = np.maximum(np.exp(-d2 / APAP_SIGMA**2), APAP_WEIGHT_FLOOR)
        weights = np.r_[weights, np.full(len(anchors), APAP_WEIGHT_FLOOR)]
        # 정규행렬의 최소 고유벡터로 weighted DLT의 해를 구한다.
        weighted = design * np.repeat(weights,2)[:,None]
        _, eigenvectors = np.linalg.eigh(weighted.T @ weighted)
        matrix = eigenvectors[:,0].reshape(3,3)
        z = matrix @ np.r_[(node-center)/scale, 1.0]
        if not np.isfinite(z).all() or abs(z[2]) < 1e-8:
            continue
        delta = z[:2]/z[2]*scale+center-node
        max_raw = max(max_raw,float(np.linalg.norm(delta)))
        delta *= np.exp(-float(d2.min()) / (2*APAP_SIGMA)**2)
        length = np.linalg.norm(delta)
        if length > APAP_MAX_CORRECTION:
            delta *= APAP_MAX_CORRECTION / length
        offsets[k] = delta
    # grid의 양 끝과 출력 pixel 좌표를 정확히 일치시키는 보간.
    mx = np.broadcast_to(np.linspace(0,nx-1,w,dtype=np.float32),(h,w)).copy()
    my = np.broadcast_to(np.linspace(0,ny-1,h,dtype=np.float32)[:,None],(h,w)).copy()
    field = cv2.remap(offsets.reshape(ny,nx,2).astype(np.float32),mx,my,cv2.INTER_LINEAR)
    valid = (mask>0).astype(np.uint8)
    padded = cv2.copyMakeBorder(valid,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    boundary_distance = cv2.distanceTransform(padded,cv2.DIST_L2,3)[1:-1,1:-1]
    field *= np.clip((boundary_distance-1.0)/32.0,0.0,1.0)[:,:,None]
    attenuation = 1.0
    for _ in range(9):
        dyx,dxx = np.gradient(field[:,:,0])
        dyy,dxy = np.gradient(field[:,:,1])
        determinant = (1+dxx)*(1+dyy)-dxy*dyx
        if float(determinant.min()) >= 0.2:
            break
        field *= 0.5
        attenuation *= 0.5
    return field, {"grid": [nx,ny], "controls": len(source),
        "max_raw_offset": max_raw, "fold_attenuation": attenuation,
        "max_applied_offset": float(np.linalg.norm(field,axis=2).max())}


def _apap_refine(warped, masks, corners, controls, *, label):
    """APAP 계열 국소 기하 보정. 노출 보정, 이음선, 블렌딩과 독립적이다."""
    output, output_masks = [], []
    for i,(im,mask,corner,(source,target)) in enumerate(zip(warped,masks,corners,controls)):
        if len(source) < MIN_INLIERS:
            raise ValueError(f"{label}: APAP image {i+1} has only {len(source)} valid controls")
        field,diagnostic = _moving_dlt_field(source,target,mask.shape,corner,mask)
        h,w = mask.shape
        mx = np.arange(w,dtype=np.float32)[None,:] + field[:,:,0]
        my = np.arange(h,dtype=np.float32)[:,None] + field[:,:,1]
        output.append(cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
        output_masks.append(cv2.remap(mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT))
        q = (target-np.asarray(corner)).astype(np.float32)
        delta = cv2.remap(field,q[:,0].reshape(-1,1),q[:,1].reshape(-1,1),cv2.INTER_LINEAR).reshape(-1,2)
        before = np.linalg.norm(source-target,axis=1)
        after = np.linalg.norm(target+delta-source,axis=1)
        diagnostic.update(stage="apap", image=i+1,
            fitting_median_before=float(np.median(before)), fitting_median_after=float(np.median(after)))
        EXPERIMENT_DIAGNOSTICS.append(diagnostic)
        _log(f"[{label}] APAP Moving DLT image {i+1}: controls={len(source)}, "
             f"max displacement={diagnostic['max_applied_offset']:.2f}px, "
             f"control inverse residual {np.median(before):.3f}->{np.median(after):.3f}px "
             f"(fitting diagnostic, not benchmark)")
    return output,output_masks


def _read_images(img_paths, expected):
    if len(img_paths) != expected:
        raise ValueError(f"Expected {expected} image paths, got {len(img_paths)}.")
    images = []
    for path in img_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        if min(h, w) < 16 or max(h, w) > MAX_SIDE:
            raise ValueError(f"Unsupported image size {w}x{h}: {path}")
        images.append(image)
    return images


def _working_images(images, work_megapix=WORK_MEGAPIX):
    """공통 축소율과 실제 반올림 크기를 반영한 원본->작업 좌표 행렬."""
    max_area = max(im.shape[0] * im.shape[1] for im in images)
    scale = min(1.0, np.sqrt(work_megapix * 1e6 / max_area))
    work, transforms = [], []
    for image in images:
        h, w = image.shape[:2]
        ww, hh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        small = image if (ww, hh) == (w, h) else cv2.resize(
            image, (ww, hh), interpolation=cv2.INTER_AREA
        )
        work.append(small)
        transforms.append(np.diag([ww / w, hh / h, 1.0]))
    return work, transforms


def _check_canvas(width, height):
    width, height = int(width), int(height)
    if (width <= 0 or height <= 0 or max(width, height) > MAX_SIDE
            or width * height > MAX_CANVAS_PIXELS):
        raise ValueError(
            f"Unsafe panorama size: {width}x{height}. "
            "Check matches / camera estimates, or resize the input images."
        )


def _union_roi(corners, sizes):
    left = min(int(c[0]) for c in corners)
    top = min(int(c[1]) for c in corners)
    right = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    bottom = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
    _check_canvas(right - left, bottom - top)
    return left, top, right - left, bottom - top


def _feather_blend(warped, masks, corners, *, return_geometry=False):
    """별도 유효 마스크의 경계 거리로 가중 합성. 검은 물체도 유효 픽셀로 보존."""
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    left, top, width, height = _union_roi(corners, sizes)
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for image, mask, corner in zip(warped, masks, corners):
        valid = (mask > 0).astype(np.uint8)
        if not np.any(valid):
            raise ValueError("An input image has no valid pixels after warping.")
        # 테두리에 0을 추가하여 직사각형 마스크에도 경계 거리가 정의되도록 함.
        padded = cv2.copyMakeBorder(valid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        weight = np.minimum(distance / FEATHER_WIDTH, 1.0) * valid
        x, y = int(corner[0]) - left, int(corner[1]) - top
        h, w = valid.shape
        accum[y:y+h, x:x+w] += image.astype(np.float32) * weight[:, :, None]
        weights[y:y+h, x:x+w] += weight

    valid = weights > 0
    np.divide(accum, np.maximum(weights, 1e-6)[:, :, None], out=accum)
    result = np.clip(np.rint(accum), 0, 255).astype(np.uint8)
    result[~valid] = 0
    # 전체가 빈 행/열만 제거. 직사각형 내부의 빈 모서리는 그대로 유지함.
    rows = np.flatnonzero(np.any(valid, axis=1))
    cols = np.flatnonzero(np.any(valid, axis=0))
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Empty panorama.")
    roi = np.s_[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    image = np.ascontiguousarray(result[roi])
    if return_geometry:
        origin = (int(left + cols[0]), int(top + rows[0]))
        return image, valid[roi].astype(np.float32), origin
    return image


def _image_corners(image):
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])


def _seam_masks(warped, masks, corners, *, cost_type="COST_COLOR",
                dilate_radius=SEAM_DILATE_RADIUS):
    """정합 결과에서 영상별 소유 영역 선택. 기하 오차 자체를 수정하지는 않음."""
    if not USE_GRAPH_CUT:
        return masks
    finder = cv2.detail_GraphCutSeamFinder(cost_type)
    seam_inputs = [cv2.UMat(mask.copy()) for mask in masks]
    seam = finder.find(
        [im.astype(np.float32) for im in warped],
        [tuple(map(int, c)) for c in corners],
        seam_inputs
    )
    # OpenCV 4.5는 입력 UMat를 수정하고 None을 반환하며, 이후 버전은 반환값도 제공.
    if seam is None:
        seam = seam_inputs
    radius = dilate_radius
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    result = []
    for selected, valid in zip(seam, masks):
        selected = selected.get() if hasattr(selected, "get") else selected
        expanded = cv2.dilate(selected, kernel)
        result.append(cv2.bitwise_and(expanded, valid))
    return result


def _compensate_exposure(warped, masks, corners, *, label="Q2"):
    """이음선 선택 전의 겹침 전체에서 부드러운 블록별 밝기 배율을 추정."""
    compensator = cv2.detail_BlocksGainCompensator(
        Q2_EXPOSURE_BLOCK, Q2_EXPOSURE_BLOCK
    )
    compensator.setNrGainsFilteringIterations(2)
    compensator.feed(corners, warped, masks)
    corrected = []
    for index, (image, mask, corner) in enumerate(zip(warped, masks, corners)):
        corrected.append(compensator.apply(index, corner, image.copy(), mask))
    _log(f"[{label}] exposure: block gain, block={Q2_EXPOSURE_BLOCK}px")
    return corrected


def _multiband_blend(warped, masks, corners, *, label="Q2"):
    """이미지, 유효 마스크, 평면 원점을 함께 반환하여 재투영 좌표계를 보존."""
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    left, top, width, height = _union_roi(corners, sizes)
    blender = cv2.detail_MultiBandBlender()
    # 작은 입력에서도 피라미드 깊이가 영상 크기를 넘지 않도록 제한.
    bands = min(Q2_BLEND_BANDS, max(1, int(np.floor(np.log2(min(width, height))))))
    blender.setNumBands(bands)
    blender.prepare((left, top, width, height))
    for image, mask, corner in zip(warped, masks, corners):
        if not np.any(mask):
            raise ValueError(f"{label}: an input has no pixels after seam selection.")
        blender.feed(image.astype(np.int16), mask, tuple(map(int, corner)))
    blended, result_mask = blender.blend(None, None)
    valid = result_mask > 0
    rows = np.flatnonzero(np.any(valid, axis=1))
    cols = np.flatnonzero(np.any(valid, axis=0))
    if not len(rows) or not len(cols):
        raise ValueError(f"{label}: empty panorama after multi-band blending.")
    result = np.clip(blended, 0, 255).astype(np.uint8)
    result[~valid] = 0
    roi = np.s_[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    origin = (int(left + cols[0]), int(top + rows[0]))
    _log(f"[{label}] blend: multi-band, bands={bands}")
    return (np.ascontiguousarray(result[roi]),
            valid[roi].astype(np.float32), origin)


def _scaled_seam_masks(warped, masks, corners, megapix):
    """공통 좌표계를 유지하며 이음선만 축소 해상도에서 찾고 원본으로 복원."""
    scale = min(1.0, np.sqrt(megapix * 1e6 / max(im.shape[0] * im.shape[1] for im in warped)))
    if scale == 1.0:
        return _seam_masks(warped, masks, corners, cost_type=Q2_SEAM_COST)
    small, small_masks, small_corners = [], [], []
    for image, mask, corner in zip(warped, masks, corners):
        corner = np.asarray(corner, dtype=np.float64)
        low = np.floor(corner * scale).astype(int)
        high = np.ceil((corner + [image.shape[1], image.shape[0]]) * scale).astype(int)
        size = tuple(map(int, high - low))
        shift = corner * scale - low
        matrix = np.float64([[scale, 0, shift[0]], [0, scale, shift[1]]])
        small.append(cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT))
        small_masks.append(cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT))
        small_corners.append(tuple(map(int, low)))
    selected = _seam_masks(small, small_masks, small_corners,
                           cost_type=Q2_SEAM_COST, dilate_radius=1)
    radius = max(Q2_SEAM_DILATE_RADIUS, int(np.ceil(1 / scale)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    restored = []
    for chosen, valid, corner, low in zip(selected, masks, corners, small_corners):
        h, w = valid.shape
        x = ((corner[0] + np.arange(w)) * scale - low[0]).astype(np.float32)
        y = ((corner[1] + np.arange(h)) * scale - low[1]).astype(np.float32)
        mx = np.ascontiguousarray(np.broadcast_to(x, (h, w)))
        my = np.ascontiguousarray(np.broadcast_to(y[:, None], (h, w)))
        mask = cv2.remap(chosen, mx, my, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        restored.append(cv2.bitwise_and(cv2.dilate(mask, kernel), valid))
    return restored


def _hard_composite(warped, masks, corners, *, label):
    """유효 마스크의 픽셀을 그대로 선택. 겹치는 영역은 뒤 입력이 우선."""
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    left, top, width, height = _union_roi(corners, sizes)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for image, mask, corner in zip(warped, masks, corners):
        valid = mask > 0
        if not np.any(valid):
            raise ValueError(f"{label}: an input has no valid pixels for composition.")
        x, y = int(corner[0]) - left, int(corner[1]) - top
        h, w = mask.shape
        region = result[y:y+h, x:x+w]
        region[valid] = image[valid]
        coverage[y:y+h, x:x+w] |= valid
    rows = np.flatnonzero(np.any(coverage, axis=1))
    cols = np.flatnonzero(np.any(coverage, axis=0))
    if not len(rows) or not len(cols):
        raise ValueError(f"{label}: empty panorama after hard composition.")
    roi = np.s_[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    origin = (int(left + cols[0]), int(top + rows[0]))
    _log(f"[{label}] hard composition: later input wins; no pixel blending")
    return np.ascontiguousarray(result[roi]), coverage[roi].astype(np.float32), origin


def _maximum_weight_masks(masks, corners):
    """경계 거리 최대인 영상에 소유권을 부여; graph-cut 없이 초기 blend mask 생성.

    Brown–Lowe의 max-weight 선택을 워핑된 마스크의 경계 거리로 근사한다.
    이 마스크는 multi-band 피라미드에서 부드러워진다.
    """
    sizes = [(mask.shape[1], mask.shape[0]) for mask in masks]
    left, top, width, height = _union_roi(corners, sizes)
    best = np.zeros((height, width), np.float32)
    owner = np.full((height, width), -1, np.int16)
    for index, (mask, corner) in enumerate(zip(masks, corners)):
        valid = (mask > 0).astype(np.uint8)
        padded = cv2.copyMakeBorder(valid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        weight = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        x, y = int(corner[0]) - left, int(corner[1]) - top
        h, w = mask.shape
        current = best[y:y+h, x:x+w]
        selected = (valid > 0) & (weight > current)
        current[selected] = weight[selected]
        owner[y:y+h, x:x+w][selected] = index
    selected_masks = []
    for index, (mask, corner) in enumerate(zip(masks, corners)):
        x, y = int(corner[0]) - left, int(corner[1]) - top
        h, w = mask.shape
        selected_masks.append(((owner[y:y+h, x:x+w] == index) * 255).astype(np.uint8))
    return selected_masks


def _brown_lowe_compose(warped, masks, corners, *, label):
    """과제 프로필: Q1/Q2/Q3/E1은 raw, E2는 global gain + multi-band."""
    if label != "E2":
        _log(f"[{label}] Brown-Lowe assignment baseline: all postprocessing OFF")
        return _hard_composite(warped, masks, corners, label=label)
    compensator = cv2.detail_GainCompensator()
    compensator.feed(corners, warped, masks)
    corrected = [compensator.apply(i, corner, image.copy(), mask)
                 for i, (image, mask, corner) in enumerate(zip(warped, masks, corners))]
    selected = _maximum_weight_masks(masks, corners)
    _log(f"[{label}] Brown-Lowe assignment baseline: global gain + max-weight masks "
         "+ multi-band; no block gain or graph-cut")
    return _multiband_blend(corrected, selected, corners, label=label)


def _compose_planar(warped, masks, corners, *, label, advanced=True, seam_megapix=None):
    """Q2/Q3는 노출/이음선/블렌딩을 한꺼번에 켜거나 끈다. E1/E2는 기존 방식."""
    if BASELINE_PROFILE == "brown_lowe_assignment":
        return _brown_lowe_compose(warped, masks, corners, label=label)
    if ((label == "Q2" and not Q2_USE_BLENDING)
            or (label == "Q3" and not Q3_USE_BLENDING)):
        _log(f"[{label}] postprocessing OFF: no exposure compensation, seam finding or blending")
        return _hard_composite(warped, masks, corners, label=label)
    corrected = (_compensate_exposure(warped, masks, corners, label=label)
                 if advanced else warped)
    if seam_megapix is None:
        selected = _seam_masks(corrected, masks, corners, cost_type=Q2_SEAM_COST,
                               dilate_radius=Q2_SEAM_DILATE_RADIUS)
    else:
        selected = _scaled_seam_masks(corrected, masks, corners, seam_megapix)
    if advanced:
        return _multiband_blend(corrected, selected, corners, label=label)
    _log(f"[{label}] blend: feather, no exposure compensation (baseline)")
    return _feather_blend(corrected, selected, corners, return_geometry=True)


def _log_alignment(features, pairwise, scales, intrinsics, rotations, warper, *, label="Q2"):
    """BA 후 인라이어의 공통 평면 좌표 오차. 최종 출력의 평가 점수는 아님."""
    if not VERBOSE:
        return
    keypoints = [feature.getKeypoints() for feature in features]
    for pair in pairwise:
        i, j = int(pair.src_img_idx), int(pair.dst_img_idx)
        if not (0 <= i < j < len(features)) or pair.num_inliers == 0:
            continue
        matches = pair.getMatches()
        keep = np.asarray(pair.getInliers()).ravel().astype(bool)
        points = [np.float32([keypoints[i][m.queryIdx].pt for m in matches])[keep],
                  np.float32([keypoints[j][m.trainIdx].pt for m in matches])[keep]]
        projected = []
        lower = np.ones(len(points[0]), dtype=bool)
        for index, pts in zip((i, j), points):
            lower &= pts[:, 1] > features[index].img_size[1] * 0.65
            original = pts / np.diag(scales[index])[:2]
            projected.append(np.float64([
                warper.warpPoint(tuple(map(float, point)),
                                 intrinsics[index], rotations[index])
                for point in original
            ]))
        errors = np.linalg.norm(projected[0] - projected[1], axis=1)
        finite = np.isfinite(errors)
        if not np.any(finite):
            _log(f"[{label}] alignment {i+1}-{j+1}: no finite projections")
            continue
        low_errors = errors[lower & finite]
        low_text = f"{np.median(low_errors):.2f}px" if len(low_errors) else "n/a"
        _log(f"[{label}] alignment {i+1}-{j+1}: "
             f"inliers={len(errors)}/{len(matches)}, "
             f"median={np.median(errors[finite]):.2f}px, "
             f"p95={np.percentile(errors[finite], 95):.2f}px, "
             f"lower65%+={low_text} (n={len(low_errors)}), planar coordinates")


def q1(img_paths):
    """Q1: 두 영상을 첫 번째 영상의 평면 좌표에 정합. BGR uint8 반환."""
    cv2.setRNGSeed(0)
    images = _read_images(img_paths, 2)
    work, scales = _working_images(images)
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    keypoints, descriptors = [], []
    for image in work:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < MIN_INLIERS:
            raise ValueError("Q1: too few SIFT features; use overlapping textured images.")
        keypoints.append(kp)
        descriptors.append(_root_descriptors(desc))

    # query = image 2, train = image 1: H21의 방향을 일관되게 유지.
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < Q1_RATIO * pair[1].distance]
    if len(good) < MIN_INLIERS:
        raise ValueError(f"Q1: only {len(good)} matches passed the ratio test.")
    src = np.float32([keypoints[1][m.queryIdx].pt for m in good])
    dst = np.float32([keypoints[0][m.trainIdx].pt for m in good])
    h_work, inliers = cv2.findHomography(
        src, dst, _homography_method(), Q1_RANSAC_PX, maxIters=5000, confidence=0.995
    )
    if h_work is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        raise ValueError("Q1: reliable homography could not be estimated.")
    # 축소 영상에서 구한 H를 원본 영상의 좌표로 복원.
    h21 = np.linalg.inv(scales[0]) @ h_work @ scales[1]
    if not np.isfinite(h21).all() or np.linalg.matrix_rank(h21) < 3:
        raise ValueError("Q1: invalid homography.")
    corners2 = _image_corners(images[1])
    homogeneous = np.column_stack([corners2, np.ones(4)]) @ h21.T
    denom = homogeneous[:, 2]
    # 사영의 무한대 선이 영상 내부를 통과하면 코너만으로 안전한 ROI를 구할 수 없음.
    if np.min(np.abs(denom)) < 1e-8 or not (np.all(denom > 0) or np.all(denom < 0)):
        raise ValueError("Q1: homography crosses the projective horizon.")
    projected = homogeneous[:, :2] / denom[:, None]
    all_corners = np.vstack([_image_corners(images[0]), projected])
    low = np.floor(all_corners.min(axis=0))
    high = np.ceil(all_corners.max(axis=0))
    width, height = map(int, high - low)
    _check_canvas(width, height)
    translation = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], np.float64)
    warped, masks = [], []
    for image, matrix in zip(images, [translation, translation @ h21]):
        warped.append(cv2.warpPerspective(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        ))
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        masks.append(cv2.warpPerspective(
            mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ))
    _log(f"[Q1] keypoints={[len(k) for k in keypoints]}, "
         f"matches={len(good)}, inliers={int(inliers.sum())}/{len(good)}")
    corners = [(0, 0), (0, 0)]
    _log(f"[Q1] algorithm={ALGORITHM}, estimator={'MAGSAC++' if 'magsac' in ALGORITHM else 'RANSAC'}")
    EXPERIMENT_DIAGNOSTICS.append({"stage":"matching", "algorithm":ALGORITHM,
        "estimator":"MAGSAC++" if "magsac" in ALGORITHM else "RANSAC",
        "candidates":len(good), "inliers":int(inliers.sum())})
    if ALGORITHM.endswith("apap"):
        keep = inliers.ravel()>0
        source = src[keep] / np.diag(scales[1])[:2]
        target = dst[keep] / np.diag(scales[0])[:2]
        projected = cv2.perspectiveTransform(source.reshape(-1,1,2),translation@h21).reshape(-1,2)
        target = cv2.perspectiveTransform(target.reshape(-1,1,2),translation).reshape(-1,2)
        # Q1의 첫 영상은 기준 평면으로 고정한다.
        refined,refined_masks = _apap_refine([warped[1]],[masks[1]],[(0,0)],
                                            [(projected,target)],label="Q1")
        warped[1],masks[1] = refined[0],refined_masks[0]
    if BASELINE_PROFILE == "brown_lowe_assignment":
        result, _, _ = _brown_lowe_compose(warped, masks, corners, label="Q1")
    else:
        result = _feather_blend(warped, _seam_masks(warped, masks, corners), corners)
    _log(f"[Q1] output={result.shape[1]}x{result.shape[0]} (planar)")
    return result


def _require_connected_graph(pairwise, count, *, label="Q2", confidence=Q2_GRAPH_CONF):
    """전체 입력이 BA에 사용할 연결 그래프에 포함되는지 확인."""
    adjacency = [set() for _ in range(count)]
    for match in pairwise:
        i, j = int(match.src_img_idx), int(match.dst_img_idx)
        if not (0 <= i < j < count):
            continue
        _log(f"[{label}] pair {i+1}-{j+1}: inliers={match.num_inliers}, "
             f"confidence={match.confidence:.3f}")
        if match.num_inliers >= MIN_INLIERS and match.confidence > confidence:
            adjacency[i].add(j)
            adjacency[j].add(i)
    visited, pending = {0}, [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    if len(visited) != count:
        raise ValueError(
            f"{label}: not all {count} images are connected by reliable matches. "
            "No images were dropped. Check overlap and pairwise match logs."
        )


def _largest_valid_rectangle(valid):
    """각 행까지 누적된 유효 높이의 히스토그램에서 최대 면적 직사각형을 탐색."""
    height = np.zeros(valid.shape[1], dtype=np.int32)
    best_area = 0
    best = None
    for row, mask in enumerate(valid):
        height += 1
        height[~mask] = 0
        stack = []
        # 같은 높이의 연속된 열은 하나의 구간으로 처리한다.
        changes = np.flatnonzero(height[1:] != height[:-1]) + 1
        for col in np.r_[0, changes, height.size].tolist():
            level = int(height[col]) if col < height.size else 0
            start = col
            while stack and stack[-1][1] > level:
                start, old = stack.pop()
                area = (col - start) * old
                if area > best_area:
                    best_area = area
                    best = (start, row + 1 - old, col, row + 1)
            if level and (not stack or stack[-1][1] < level):
                stack.append((start, level))
    if best is None:
        raise ValueError("Q2: no valid rectangle after reprojection.")
    return best


def _cylindrical_panorama(planar, valid, origin, scale, *, label="Q2"):
    """합성이 끝난 평면 파노라마 전체를 원통 좌표로 한 번 재투영.

    평면 좌표 x, y는 광축 기준이며 origin은 크롭된 평면 영상의 좌상단이다.
    정방향: u = f * atan(x/f), v = y / sqrt(1 + (x/f)**2).
    역방향: x = f * tan(u/f), y = v / cos(u/f).
    f는 평면 합성에 사용한 warp scale과 같아야 한다.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: invalid cylindrical projection scale.")
    valid = np.asarray(valid, dtype=np.float32)
    if valid.shape != planar.shape[:2] or not np.any(valid > 0):
        raise ValueError(f"{label}: empty or mismatched planar validity mask.")
    left, top = origin
    h, w = valid.shape
    # 배열 중심이 아닌, 평면 투영에서 정의한 광축을 유지한다.
    angles = np.arctan((left + np.arange(w, dtype=np.float64)) / scale)
    u = scale * angles
    cosines = np.cos(angles)
    # 열별 유효 영역의 상/하단에서 출력 경계를 계산한다.
    support = valid > 0
    occupied = support.any(axis=0)
    upper = (top + support.argmax(axis=0)) * cosines
    lower = (top + h - 1 - support[::-1].argmax(axis=0)) * cosines
    u0, u1 = int(np.floor(u[occupied].min())), int(np.ceil(u[occupied].max()))
    v0, v1 = int(np.floor(upper[occupied].min())), int(np.ceil(lower[occupied].max()))
    _check_canvas(u1 - u0 + 1, v1 - v0 + 1)

    theta = np.arange(u0, u1 + 1, dtype=np.float64) / scale
    if np.any(np.abs(theta) >= np.pi / 2):
        raise ValueError(f"{label}: cylindrical reprojection exceeds the planar field of view.")
    src_x = (scale * np.tan(theta) - left).astype(np.float32)
    row_v = np.arange(v0, v1 + 1, dtype=np.float32)[:, None]
    map_y = (row_v / np.cos(theta).astype(np.float32) - top).astype(np.float32)
    map_x = np.ascontiguousarray(np.broadcast_to(src_x, map_y.shape))
    # 원통 재투영 결과 전체를 유지하고, 유효 영역 밖은 검은색으로 표시한다.
    coverage = cv2.remap(valid, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = cv2.remap(planar, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[coverage <= 0] = 0
    _log(f"[{label}] cylindrical canvas={u1-u0+1}x{v1-v0+1}, "
         "no crop (full reprojected image)")
    return result


def _camera_planar_mosaic(img_paths, *, expected, label, work_megapix,
                          match_conf, graph_conf, grid_cols=None, advanced=True):
    """모든 입력의 카메라 관계를 추정하고 공통 평면 모자이크 한 장을 반환."""
    cv2.setRNGSeed(0)
    images = _read_images(img_paths, expected)
    work, scales = _working_images(images, work_megapix)
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    features = []
    for index, image in enumerate(work):
        feature = cv2.detail.computeImageFeatures2(sift, image)
        feature.img_idx = index
        if "rootsift" in ALGORITHM:
            feature.descriptors = cv2.UMat(_root_descriptors(feature.descriptors.get()))
        if len(feature.getKeypoints()) < MIN_INLIERS:
            raise ValueError(f"{label}: image {index+1} has too few SIFT features.")
        features.append(feature)
    _log(f"[{label}] keypoints={[len(f.getKeypoints()) for f in features]}")

    matcher = cv2.detail_BestOf2NearestMatcher(False, match_conf)
    pairwise = (matcher.apply2(features) if grid_cols is None else
                matcher.apply2(features, _grid_pair_mask(expected, grid_cols)))
    matcher.collectGarbage()
    native_pairwise = pairwise
    if ALGORITHM != "brown_lowe":
        pairwise = _experiment_matches(features, native_pairwise, label=label)
    # leaveBiggestComponent를 호출하지 않음: 입력 일부 누락을 성공으로 처리하지 않음.
    _require_connected_graph(pairwise, len(images), label=label, confidence=graph_conf)

    estimator = cv2.detail_HomographyBasedEstimator()
    ok, cameras = estimator.apply(features, native_pairwise, None)
    if not ok or len(cameras) != len(images):
        raise ValueError(f"{label}: initial camera estimation failed.")
    for camera in cameras:
        camera.R = camera.R.astype(np.float32)
        if not np.isfinite(camera.focal) or camera.focal <= 0:
            raise ValueError(f"{label}: initial focal estimate is invalid.")

    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(graph_conf)
    # Ray adjuster는 초점거리와 회전을 최적화하며, refinement mask는 사용하지 않음.
    # 주점은 작업 영상의 중심에 유지됨.
    if ALGORITHM == "brown_lowe":
        ok, cameras = adjuster.apply(features, pairwise, cameras)
    else:
        cameras = _ray_bundle_adjust(features, pairwise, cameras, graph_conf, label=label)
        ok = True
    if not ok:
        raise ValueError(f"{label}: ray bundle adjustment failed.")
    for camera in cameras:
        if (not np.isfinite(camera.K()).all() or not np.isfinite(camera.R).all()
                or camera.focal <= 0):
            raise ValueError(f"{label}: optimized camera parameters are invalid.")

    rotations = cv2.detail.waveCorrect(
        [camera.R.astype(np.float32).copy() for camera in cameras],
        cv2.detail.WAVE_CORRECT_HORIZ
    )
    # Q2는 가운데 카메라, Q3는 전체 카메라의 평균 방향으로 평면 광축을 정함.
    axis = (rotations[1][:, 2] if expected == 3 else
            np.mean([rotation[:, 2] for rotation in rotations], axis=0))
    yaw = np.arctan2(axis[0], axis[2])
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_correction = np.float32([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    rotations = [(yaw_correction @ r).astype(np.float32) for r in rotations]

    # 원본 크기로 합성: 실제 resize 비율을 사용하여 fx, fy, cx, cy를 복원.
    intrinsics = [(np.linalg.inv(scale) @ camera.K()).astype(np.float32)
                  for scale, camera in zip(scales, cameras)]
    warp_scale = float(np.median([k[0, 0] for k in intrinsics]))
    if not np.isfinite(warp_scale) or warp_scale <= 0:
        raise ValueError(f"{label}: invalid planar warp scale.")
    _log(f"[{label}] focal_px={[round(float(k[0, 0]), 2) for k in intrinsics]}")
    # 모든 입력에 같은 평면 투영기와 광축을 사용한다.
    plane_warper = cv2.PyRotationWarper("plane", warp_scale)
    corners, sizes = [], []
    for image, k, rotation in zip(images, intrinsics, rotations):
        # 평면의 무한대 선을 가로지르거나 카메라 뒤쪽에 있는 입력은 거부.
        # z는 입력 좌표의 선형 함수이므로 네 코너의 검사로 전체를 판별 가능.
        points = np.column_stack([_image_corners(image), np.ones(4)])
        rays = points @ (rotation @ np.linalg.inv(k)).T
        if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 1e-6):
            raise _PlanarProjectionError(
                f"{label}: input crosses the planar projection horizon. "
                "A single planar panorama requires a field of view below 180 degrees.",
                images, intrinsics, rotations, warp_scale,
                alignment=(features,pairwise,scales,graph_conf)
            )
        roi = plane_warper.warpRoi((image.shape[1], image.shape[0]), k, rotation)
        _check_canvas(roi[2], roi[3])
        corners.append(roi[:2])
        sizes.append(roi[2:])
    _union_roi(corners, sizes)  # 큰 배열을 할당하기 전에 전체 출력 크기 검증
    _log_alignment(features, pairwise, scales, intrinsics, rotations, plane_warper, label=label)

    warped, masks, actual_corners = [], [], []
    for image, k, rotation in zip(images, intrinsics, rotations):
        corner, warped_image = plane_warper.warp(
            image, k, rotation, cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        )
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        mask_corner, warped_mask = plane_warper.warp(
            mask, k, rotation, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT
        )
        if corner != mask_corner or warped_image.shape[:2] != warped_mask.shape:
            raise ValueError(f"{label}: warped image and mask coordinates differ.")
        actual_corners.append(corner)
        warped.append(warped_image)
        masks.append(warped_mask)
    if ALGORITHM.endswith("apap"):
        controls = _apap_tracks(features,pairwise,scales,intrinsics,rotations,plane_warper,graph_conf)
        warped,masks = _apap_refine(warped,masks,actual_corners,controls,label=label)
    planar, valid, origin = _compose_planar(
        warped, masks, actual_corners, label=label, advanced=advanced
    )
    _log(f"[{label}] planar mosaic={planar.shape[1]}x{planar.shape[0]} ({expected}/{expected} images)")
    return planar, valid, origin, warp_scale


def _q2_planar_mosaic(img_paths):
    """Q2: 세 입력을 공통 평면에서 완전히 합성한 뒤 반환."""
    return _camera_planar_mosaic(
        img_paths, expected=3, label="Q2", work_megapix=Q2_WORK_MEGAPIX,
        match_conf=Q2_MATCH_CONF, graph_conf=Q2_GRAPH_CONF
    )


def q2(img_paths):
    """3장 -> 평면 파노라마 1장 -> 하나의 원통 좌표계. BGR uint8 반환.

    1단계에서 노출 보정, 이음선 선택, 옵션에 따른 합성까지 평면에서 끝낸다.
    2단계는 완성된 이미지 한 장만 입력받아 원통 재투영을 한 번 수행한다.
    원통 재투영 후에는 다시 정합하거나 블렌딩하지 않으며 크롭하지 않는다.
    """
    planar, valid, origin, warp_scale = _q2_planar_mosaic(img_paths)
    _log("[Q2] reprojecting ONE completed planar mosaic to ONE cylinder")
    result = _cylindrical_panorama(planar, valid, origin, warp_scale)
    _log(f"[Q2] output={result.shape[1]}x{result.shape[0]} "
         "(planar stitching -> cylindrical reprojection, 3/3 images)")
    return result


def _grid_pair_mask(count, columns):
    """Raster 순서 격자의 가로/세로/대각 이웃만 매칭. 행 끝과 다음 행 시작은 제외."""
    mask = np.zeros((count, count), dtype=np.uint8)
    for i in range(count):
        for j in range(i + 1, count):
            if max(abs(i // columns - j // columns),
                   abs(i % columns - j % columns)) <= 1:
                mask[i, j] = mask[j, i] = 1
    return mask


def _q3_translation_mosaic(images, *, label="Q3", advanced=True):
    """이동만으로 설명되는 입력을 자동 검출해 전역 이동을 공동 추정.

    회전/원근 변화를 억지로 이동 모델에 맞추지 않는다. 조건을 만족하지
    않으면 None을 반환하여 카메라 모델로 진행한다.
    """
    work, scales = _working_images(images, Q3_WORK_MEGAPIX)
    # 서로 다른 축척의 입력에는 원본 좌표의 단순 이동을 가정할 수 없다.
    if any(not np.allclose(scale, scales[0]) for scale in scales):
        return None
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
                for im in work]
    if any(desc is None or len(kp) < MIN_INLIERS for kp, desc in features):
        return None
    _log(f"[{label}] keypoints={[len(kp) for kp, _ in features]}")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = _grid_pair_mask(len(images), 4)
    edges = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if not neighbors[i, j]:
                continue
            pairs = matcher.knnMatch(features[i][1], features[j][1], k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2
                    and pair[0].distance < Q3_RATIO * pair[1].distance]
            if len(good) < MIN_INLIERS:
                continue
            src = np.float32([features[i][0][m.queryIdx].pt for m in good])
            dst = np.float32([features[j][0][m.trainIdx].pt for m in good])
            h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
            if h is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
                continue
            if np.mean(inliers) < 0.5:
                continue
            keep = inliers.ravel() > 0
            differences = src[keep] - dst[keep]
            offset = np.median(differences, axis=0)
            residual = np.linalg.norm(differences - offset, axis=1)
            if np.percentile(residual, 95) > Q3_TRANSLATION_P95:
                _log(f"[{label}] pair {i+1}-{j+1}: translation inadequate; use camera model")
                return None
            edges.append((i, j, offset, differences))
    if not edges:
        return None

    # 위치 j - 위치 i = 대응점 i - 대응점 j. 0번을 임시 원점으로 두고 전체 관계를 함께 푼다.
    design = np.zeros((len(edges), len(images) - 1), dtype=np.float64)
    targets, weights = [], []
    for row, (i, j, offset, differences) in enumerate(edges):
        if i:
            design[row, i - 1] = -1
        design[row, j - 1] = 1
        targets.append(offset)
        weights.append(np.sqrt(min(len(differences), 200)))
    if np.linalg.matrix_rank(design) != len(images) - 1:
        _log(f"[{label}] translation graph disconnected; use camera model")
        return None
    weights = np.asarray(weights)[:, None]
    solved = np.linalg.lstsq(design * weights, np.asarray(targets) * weights, rcond=None)[0]
    positions = np.vstack([np.zeros(2), solved])
    residuals = []
    for i, j, _, differences in edges:
        errors = np.linalg.norm(differences - (positions[j] - positions[i]), axis=1)
        if np.percentile(errors, 95) > Q3_TRANSLATION_P95:
            _log(f"[{label}] translation loop inconsistency; use camera model")
            return None
        residuals.extend(errors.tolist())
        _log(f"[{label}] pair {i+1}-{j+1}: inliers={len(errors)}, "
             f"median={np.median(errors):.3f}px (work coordinates)")
    _log(f"[{label}] global translation: 8/8 images, "
         f"median={np.median(residuals):.3f}px, p95={np.percentile(residuals, 95):.3f}px")

    positions /= np.diag(scales[0])[:2]
    sizes = np.array([(im.shape[1], im.shape[0]) for im in images])
    low = positions.min(axis=0)
    high = (positions + sizes).max(axis=0)
    _check_canvas(*np.ceil(high - low + 2))
    # 회전 모델에서 식별할 수 없는 광축은 전체 평면 영역의 중심으로 정의한다.
    positions -= (low + high) / 2
    warped, masks, corners = [], [], []
    for image, position in zip(images, positions):
        corner = np.floor(position).astype(int)
        w, h = image.shape[1], image.shape[0]
        extent = np.ceil(position + [w, h]).astype(int) - corner
        transform = np.float64([[1, 0, position[0] - corner[0]],
                                [0, 1, position[1] - corner[1]]])
        size = tuple(map(int, extent))
        warped.append(cv2.warpAffine(image, transform, size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        masks.append(cv2.warpAffine(np.full((h, w), 255, np.uint8), transform, size,
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))
        corners.append(tuple(map(int, corner)))
    planar, valid, origin = _compose_planar(
        warped, masks, corners, label=label, advanced=advanced
    )
    # 순수 이동만으로 실제 카메라 focal을 결정할 수 없으므로 표시용 축척을 명시.
    focal = float((high[0] - low[0]) * Q3_FOCAL_WIDTH_FACTOR)
    _log(f"[{label}] focal not identifiable from translation; display scale={focal:.2f}px")
    return planar, valid, origin, focal


def _q3_planar_mosaic(img_paths, *, label="Q3", advanced=True):
    """2x4 입력의 모든 영상을 포함한 단일 평면 모자이크."""
    cv2.setRNGSeed(0)
    images = _read_images(img_paths, 8)
    result = (None if BASELINE_PROFILE == "brown_lowe_assignment" else
              _q3_translation_mosaic(images, label=label, advanced=advanced))
    if result is None:
        result = _camera_planar_mosaic(
            img_paths, expected=8, label=label, work_megapix=Q3_WORK_MEGAPIX,
            match_conf=Q3_MATCH_CONF, graph_conf=Q3_GRAPH_CONF, grid_cols=4,
            advanced=advanced
        )
    planar, _, _, _ = result
    _log(f"[{label}] planar mosaic={planar.shape[1]}x{planar.shape[0]} (8/8 images)")
    return result


def _wide_surface_mosaic(geometry, *, projection, label, advanced=True):
    """단일 평면에 담을 수 없는 8장을 하나의 공통 원통/구면 좌표계에서 합성."""
    if projection not in ("cylindrical", "spherical"):
        raise ValueError(f"Unsupported wide projection: {projection}")
    warper = cv2.PyRotationWarper(projection, geometry.scale)
    corners, sizes = [], []
    for image, k, rotation in zip(geometry.images, geometry.intrinsics, geometry.rotations):
        roi = warper.warpRoi((image.shape[1], image.shape[0]), k, rotation)
        _check_canvas(roi[2], roi[3])
        corners.append(roi[:2])
        sizes.append(roi[2:])
    _union_roi(corners, sizes)
    warped, masks, actual_corners = [], [], []
    for image, k, rotation in zip(geometry.images, geometry.intrinsics, geometry.rotations):
        corner, result = warper.warp(image, k, rotation, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask_corner, mask = warper.warp(np.full(image.shape[:2], 255, np.uint8),
                                       k, rotation, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        if corner != mask_corner or result.shape[:2] != mask.shape:
            raise ValueError(f"{label}: {projection} image and mask coordinates differ.")
        warped.append(result)
        masks.append(mask)
        actual_corners.append(corner)
    if ALGORITHM.endswith("apap"):
        features,pairwise,scales,graph_conf = geometry.alignment
        controls = _apap_tracks(features,pairwise,scales,geometry.intrinsics,
                                geometry.rotations,warper,graph_conf)
        warped,masks = _apap_refine(warped,masks,actual_corners,controls,label=label)
    result, _, _ = _compose_planar(
        warped, masks, actual_corners, label=label, advanced=advanced,
        seam_megapix=Q3_WIDE_SEAM_MEGAPIX
    )
    _log(f"[{label}] output={result.shape[1]}x{result.shape[0]} "
         f"(shared {projection} coordinates, 8/8 images, no planar reprojection)")
    return result


def _wide_cylindrical_mosaic(geometry):
    """넓은 시야의 Q3 입력을 공통 원통에서 합성."""
    return _wide_surface_mosaic(geometry, projection="cylindrical", label="Q3")


def q3(img_paths):
    """8장 원통 정합. 평면 합성 가능 시 기존 경로, 지평선을 넘으면 공통 원통 합성."""
    try:
        planar, valid, origin, scale = _q3_planar_mosaic(img_paths)
    except _PlanarProjectionError as geometry:
        _log("[Q3] field of view exceeds the common plane; switching to ONE shared cylinder")
        return _wide_cylindrical_mosaic(geometry)
    _log("[Q3] reprojecting ONE completed planar mosaic to ONE cylinder")
    return _cylindrical_panorama(planar, valid, origin, scale, label="Q3")


def _spherical_panorama(planar, valid, origin, scale, *, label):
    """완성된 평면 파노라마 한 장을 공통 구면의 경도/위도 좌표로 변환.

    광축 기준 평면 좌표를 (x, y), 공통 초점거리를 f라 하면:
      u = f * atan(x/f), v = f * atan(y / sqrt(x*x + f*f))
    역방향: x = f * tan(u/f), y = f * tan(v/f) / cos(u/f).
    OpenCV spherical 좌표의 수직 원점 f*pi/2만 생략한 동일한 투영이다.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: invalid spherical projection scale.")
    valid = np.asarray(valid, dtype=np.float32)
    if valid.shape != planar.shape[:2] or not np.any(valid > 0):
        raise ValueError(f"{label}: empty or mismatched planar validity mask.")
    left, top = origin
    h, w = valid.shape
    angles = np.arctan((left + np.arange(w, dtype=np.float64)) / scale)
    u = scale * angles
    cosines = np.cos(angles)
    support = valid > 0
    occupied = support.any(axis=0)
    # 열마다 v는 y의 단조 함수이므로 위/아래 유효 픽셀로 경계를 결정한다.
    upper_y = top + support.argmax(axis=0)
    lower_y = top + h - 1 - support[::-1].argmax(axis=0)
    upper_v = scale * np.arctan(upper_y * cosines / scale)
    lower_v = scale * np.arctan(lower_y * cosines / scale)
    u0, u1 = int(np.floor(u[occupied].min())), int(np.ceil(u[occupied].max()))
    v0, v1 = int(np.floor(upper_v[occupied].min())), int(np.ceil(lower_v[occupied].max()))
    _check_canvas(u1 - u0 + 1, v1 - v0 + 1)
    theta = np.arange(u0, u1 + 1, dtype=np.float64) / scale
    phi = np.arange(v0, v1 + 1, dtype=np.float64) / scale
    if np.any(np.abs(theta) >= np.pi / 2) or np.any(np.abs(phi) >= np.pi / 2):
        raise ValueError(f"{label}: spherical reprojection exceeds the planar field of view.")
    src_x = (scale * np.tan(theta) - left).astype(np.float32)
    map_y = (scale * np.tan(phi[:, None]) / np.cos(theta)[None, :] - top).astype(np.float32)
    map_x = np.ascontiguousarray(np.broadcast_to(src_x, map_y.shape))
    coverage = cv2.remap(valid, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result = cv2.remap(planar, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[coverage <= 0] = 0
    _log(f"[{label}] spherical canvas={u1-u0+1}x{v1-v0+1}, "
         "no crop (full reprojected image)")
    return result


def _spherical_stitch(img_paths, *, label, advanced):
    """8장 구면 정합. 좁은 시야는 평면 재투영, 넓은 시야는 공통 구면 합성."""
    try:
        planar, valid, origin, scale = _q3_planar_mosaic(
            img_paths, label=label, advanced=advanced
        )
    except _PlanarProjectionError as geometry:
        _log(f"[{label}] field of view exceeds the common plane; switching to ONE shared sphere")
        return _wide_surface_mosaic(
            geometry, projection="spherical", label=label, advanced=advanced
        )
    _log(f"[{label}] reprojecting ONE completed planar mosaic to ONE sphere")
    return _spherical_panorama(planar, valid, origin, scale, label=label)


def e1(img_paths):
    """8장의 공통 구면 정합 + 기본 feather 블렌딩."""
    return _spherical_stitch(img_paths, label="E1", advanced=False)


def e2(img_paths):
    """8장의 공통 구면 정합 + 노출 보정 + multi-band 블렌딩."""
    return _spherical_stitch(img_paths, label="E2", advanced=True)
