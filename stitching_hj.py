"""이미지 스티칭 과제: Q1 / Q2 / Q3 / E1 / E2.

baseline = Brown & Lowe 2007 (팀원 IS_ws 의 B0 와 같은 코드).
최종 모델:
  Q1 ⑤  SIFT + 엣지 교점 대응점 -> homography -> REW
  Q2 ④  원통 투영 -> SIFT -> homography -> REW
  Q3 ⑦  SIFT + 국소 일관성 매칭 -> BA -> 원통 -> DIS 대응점 + 중간점 TPS
  E1 ⑦  Q3 과 같고 구면 투영
  E2 ⑤  E1 정합(구면) + gain·비네팅 보정 + 어긋남 회피 seam + multi-band
Q1~E1 은 블렌딩 없이 입력 순서대로 덮어씀. Stitcher API 는 사용하지 않음.

REW: Li et al., TMM 2018 / BA: Brown & Lowe, IJCV 2007.
q1~e2 는 img_paths 에서만 읽고 배열을 반환. 저장·평가·벤치마크는 맨 아래 CLI
(사진은 results/, 표는 docs/hj_*.md, 벤치마크 데이터는 data/benchmarks/):
  python stitching_hj.py --task list | stitch | evaluate | benchmark --case Q3-7 ...
"""

if __name__ == "__main__":
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import math

import cv2
import numpy as np


# ================================================================ 0. 방법 선택
METHOD = {"Q1": 5, "Q2": 4, "Q3": 7, "E1": 7, "E2": 5}   # 1 = baseline, 기본값 = 최종 모델

# 방법 번호별 구성 (문서의 ①~⑦)
_BA_METHODS = {
    1: dict(name="① SIFT + BA (baseline)", kind="baseline"),
    2: dict(name="② SIFT+AKAZE + BA", kind="ba", features="sift_akaze", rew=False),
    3: dict(name="③ SIFT+AKAZE + BA+REW", kind="ba", features="sift_akaze", rew=True),
    4: dict(name="④ SIFT + BA+REW", kind="ba", features="sift", rew=True),
    5: dict(name="⑤ SIFT+엣지 + BA+REW", kind="ba", features="sift_edge", rew=True),
    6: dict(name="⑥ SIFT+DIS + BA+REW", kind="ba", features="sift", rew=True, dense=True),
    7: dict(name="⑦ SIFT+국소일관성+DIS + BA+REW", kind="ba", local=True, rew=True, dense=True),
}
METHODS = {
    "Q1": {1: dict(name="① SIFT + 호모그래피 (baseline)", kind="baseline"),
           2: dict(name="② SIFT+AKAZE + 호모그래피", kind="homography", features="sift_akaze"),
           3: dict(name="③ SIFT+AKAZE + REW", kind="rew", features="sift_akaze"),
           4: dict(name="④ SIFT + REW", kind="rew", features="sift"),
           5: dict(name="⑤ SIFT+엣지 + REW", kind="rew", features="sift_edge")},
    "Q2": {1: dict(name="① SIFT + BA (baseline)", kind="baseline"),
           2: dict(name="② SIFT+AKAZE + BA", kind="ba", features="sift_akaze", rew=False),
           3: dict(name="③ SIFT+AKAZE + REW", kind="rew", features="sift_akaze"),
           4: dict(name="④ SIFT + REW", kind="rew", features="sift"),
           5: dict(name="⑤ SIFT+엣지 + REW", kind="rew", features="sift_edge")},
    "Q3": dict(_BA_METHODS),
    "E1": dict(_BA_METHODS),
    "E2": {1: dict(name="① baseline (gain + 가중치 최대 + 멀티밴드)", kind="baseline"),
           3: dict(name="③ ⑦ 정합 + gain + 멀티밴드", kind="e2", stage=3),
           4: dict(name="④ + 비네팅 보정", kind="e2", stage=4),
           5: dict(name="⑤ + 어긋남 회피 seam (최종)", kind="e2", stage=5)},
}
PROJECTION = {"Q1": "planar", "Q2": "cylindrical", "Q3": "cylindrical", "E1": "spherical", "E2": "spherical"}

# ---------------------------------------------------------------- 우리 모델 설정
FEATURES = "sift"          # sift / sift_akaze / sift_edge (방법 번호에 따라 바뀜)
FEAT_MAX_SIDE = 1000      # 특징점 검출용 축소 크기 (긴 변)
COMPOSE_MAX_SIDE = 1400   # 합성용 이미지 크기 (긴 변)
MAX_PANO_SIDE = 8000
MAX_PANO_PIXELS = 40e6
RATIO = 0.75              # ratio test
H_THRESH = 4.0            # RANSAC 허용 오차 (px)
RANSAC_ITERS = 500


# REW
TPS_LAMBDA = 0.01         # thin-plate spline 평활 계수 (좌표를 이미지 긴 변으로 정규화한 단위)
TPS_MAX_CTRL = 800        # TPS 제어점 최대 수
TPS_GRID = 8              # 변형량을 계산하는 격자 간격 (px), 사이는 선형 보간
REW_OUT_SIGMA = 3.0       # 대응점 제거 기준: 잔차 > max(REW_OUT_MIN, 3 * 1.4826 * MAD)
REW_OUT_MIN = 2.0
REW_DECAY = 0.5           # 겹침 영역에서 이미지 폭 * 0.5 만큼 멀어지면 변형 0


# ================================================================ 공통
def _read(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # 한글 경로 등으로 imread 가 실패하는 경우
        try:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
    return img


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def load_images(img_paths, max_side=COMPOSE_MAX_SIDE):
    imgs, ids = [], []
    for k, p in enumerate(img_paths):
        im = _read(p)
        if im is not None and im.size > 0:
            imgs.append(_resize_max(im, max_side))
            ids.append(k)
    return imgs, ids


def _pt(H, p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    z = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
    return q[:, :2] / z[:, None]


def _akaze():
    for mod in (cv2, getattr(cv2, "xfeatures2d", None)):   # OpenCV 4.x: cv2, 5.x: cv2.xfeatures2d
        fn = getattr(mod, "AKAZE_create", None) if mod is not None else None
        if fn is not None:
            return fn()
    return None


def detect(img):
    """SIFT (FEATURES = "sift_akaze" 면 AKAZE 도), 특징점 수 상한 없음"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = _resize_max(gray, FEAT_MAX_SIDE)
    s = small.shape[1] / float(gray.shape[1])
    out = {}
    dets = [("SIFT", cv2.SIFT_create(nfeatures=0), cv2.NORM_L2)]                  # nfeatures=0: 상한 없음
    if FEATURES == "sift_akaze":                                                   # 기법별로 따로 매칭 후 좌표 쌍을 합침
        dets.append(("AKAZE", _akaze(), cv2.NORM_HAMMING))
    for name, det, norm in dets:
        if det is None:
            continue
        kps, des = det.detectAndCompute(small, None)
        if des is not None and len(kps) >= 4:
            out[name] = (np.float32([k.pt for k in kps]) / s, des, norm)
    return out


# ================================================================ 2. 매칭 (baseline 그대로)
def match(fa, fb):
    """기법별로 따로 매칭 후 좌표 쌍만 합친다 (SIFT 기술자와 AKAZE 기술자는 서로 비교 불가)"""
    pa_all, pb_all = [], []
    for name in fa:
        if name not in fb:
            continue
        pa, da, norm = fa[name]
        pb, db, _ = fb[name]
        if len(pa) < 2 or len(pb) < 2:
            continue
        try:
            knn = cv2.BFMatcher(norm).knnMatch(da, db, k=2)
        except cv2.error:
            continue
        ia, ib = [], []
        for pr in knn:
            if len(pr) == 2 and pr[0].distance < RATIO * pr[1].distance:
                ia.append(pr[0].queryIdx)
                ib.append(pr[0].trainIdx)
        if ia:
            pa_all.append(pa[ia])
            pb_all.append(pb[ib])
    if not pa_all:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.vstack(pa_all), np.vstack(pb_all)


def bl_homography(src, dst):
    if len(src) < 4:
        return None, None
    H, m = cv2.findHomography(src, dst, cv2.RANSAC, H_THRESH, maxIters=RANSAC_ITERS, confidence=0.999)
    if H is None or m is None:
        return None, None
    return H, m.ravel().astype(bool)


def enough_inliers(n_inl, n_match):
    """Brown & Lowe (2007) 판정식: inlier > 8 + 0.3 * 매칭 수"""
    return n_inl >= 8 + 0.3 * n_match


def focal_from_homography(H):
    """회전만 있는 두 카메라 사이 homography 로부터 초점거리 추정 (Szeliski & Shum 1997)."""
    if H is None or abs(H[2, 2]) < 1e-12:
        return None
    h = (H / H[2, 2]).ravel()

    def solve(d1, d2, v1, v2):
        if v1 < v2:
            v1, v2 = v2, v1
        if v1 > 0 and v2 > 0:
            return math.sqrt(v1 if abs(d1) > abs(d2) else v2)
        if v1 > 0:
            return math.sqrt(v1)
        return None

    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if d1 != 0 else -1.0
    v2 = (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2 if d2 != 0 else -1.0
    f1 = solve(d1, d2, v1, v2)
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    v1 = -h[2] * h[5] / d1 if d1 != 0 else -1.0
    v2 = (h[5] ** 2 - h[2] ** 2) / d2 if d2 != 0 else -1.0
    f0 = solve(d1, d2, v1, v2)
    if f0 and f1:
        return math.sqrt(f0 * f1)
    return None


# ================================================================ 2-1. 엣지 교점 대응점 (Multi-feature image stitching, Yang et al. 2025 아이디어 단순화)
# 매칭 선분과 Canny 엣지의 교점을 상대 사진에서 찾아 대응점으로 추가 (REW 제어점)
EDGE_K = 4                 # 점마다 선을 그을 상대 점 수 (거리 범위 안에서 고르게)
EDGE_D = (20.0, 300.0)     # 선분 길이 범위 (px)
EDGE_SEARCH = 3.0          # 상대 사진에서 예측 위치 주변 탐색 범위 (px)
EDGE_NCC = 0.9             # 밝기 단면 상관 기준
EDGE_HALF = 7              # 단면 길이 (±px)
EDGE_GRID = 8              # 중복 제거 격자 (px)
EDGE_ANCHORS = 400         # 선을 긋는 기준점 수 (대응점이 많으면 고르게 골라 씀)


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _profile(g, p, d, s):
    """점 p 에서 단위방향 d 로 s(배열, px) 만큼 떨어진 곳의 밝기"""
    return _bilinear(g, p[0] + d[0] * s, p[1] + d[1] * s)


def _peak(prof, idx):
    """1차 미분 크기의 부분 픽셀 위치 (포물선 맞춤)"""
    if idx <= 0 or idx >= len(prof) - 1:
        return float(idx)
    a, b, c = prof[idx - 1], prof[idx], prof[idx + 1]
    den = a - 2 * b + c
    return idx + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)


def edge_correspondences(img_a, img_b, P, Q):
    """P (사진 a), Q (사진 b): 대응점 (N x 2). 반환: 새 대응점 (A, B)"""
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    ea = cv2.Canny(cv2.GaussianBlur(ga, (3, 3), 0), 50, 150) > 0
    ga = cv2.GaussianBlur(ga, (3, 3), 0).astype(np.float32); gb = cv2.GaussianBlur(gb, (3, 3), 0).astype(np.float32)
    n = len(P)
    if n < 3:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if n > EDGE_ANCHORS:                                  # 기준점을 영상 전체에 고르게
        cell = np.round(P / 40).astype(int); _, keep = np.unique(cell[:, 0] * 100000 + cell[:, 1], return_index=True)
        if len(keep) > EDGE_ANCHORS:
            keep = keep[np.linspace(0, len(keep) - 1, EDGE_ANCHORS).astype(int)]
        P, Q = P[keep], Q[keep]; n = len(P)
    dist = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    rng = np.random.default_rng(0)
    nb = []
    for i in range(n):                                   # 거리 범위 안의 점들 중 거리 순으로 고르게 K 개
        cand = np.flatnonzero((dist[i] >= EDGE_D[0]) & (dist[i] <= EDGE_D[1]))
        if len(cand) > EDGE_K:
            cand = cand[np.argsort(dist[i, cand])][np.linspace(0, len(cand) - 1, EDGE_K).astype(int)]
        nb.append(cand)
    seen = set()
    A, B = [], []
    S = np.arange(-EDGE_HALF, EDGE_HALF + 0.5, 0.5)
    for i in range(n):
        for j in nb[i]:
            if (j, i) in seen:
                continue
            seen.add((i, j))
            p1, p2, q1, q2 = P[i], P[j], Q[i], Q[j]
            La, Lb = np.linalg.norm(p2 - p1), np.linalg.norm(q2 - q1)
            if not (EDGE_D[0] <= La <= EDGE_D[1]) or Lb < 1:
                continue
            da, db = (p2 - p1) / La, (q2 - q1) / Lb
            ts = np.arange(3, La - 3, 0.5)                       # 끝점 근처는 제외
            xs, ys = p1[0] + da[0] * ts, p1[1] + da[1] * ts
            hit = ea[np.round(ys).astype(int).clip(0, ea.shape[0] - 1), np.round(xs).astype(int).clip(0, ea.shape[1] - 1)]
            if not hit.any():
                continue
            prof_a = _profile(ga, p1, da, ts)
            der_a = np.gradient(prof_a)
            # 엣지 픽셀 묶음마다 기울기 최대 위치 하나
            idx = np.flatnonzero(hit)
            groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
            for gidx in groups:
                k = gidx[np.argmax(np.abs(der_a[gidx]))]
                ta = ts[0] + 0.5 * _peak(np.abs(der_a), k)
                sign = np.sign(der_a[k])
                if abs(der_a[k]) < 4:                            # 약한 엣지 제외
                    continue
                a = p1 + da * ta
                tb0 = ta / La * Lb                               # 국소 아핀 가정: 선분 위 비율 유지
                sb = np.arange(tb0 - EDGE_SEARCH, tb0 + EDGE_SEARCH + 0.25, 0.5)
                prof_b = _profile(gb, q1, db, sb)
                der_b = np.gradient(prof_b)
                kb = int(np.argmax(np.abs(der_b) * (np.sign(der_b) == sign)))
                if np.sign(der_b[kb]) != sign or abs(der_b[kb]) < 4:
                    continue
                tb = sb[0] + 0.5 * _peak(np.abs(der_b), kb)
                b = q1 + db * tb
                pa_ = _profile(ga, a, da, S); pb_ = _profile(gb, b, db, S * Lb / La)
                pa_ = pa_ - pa_.mean(); pb_ = pb_ - pb_.mean()
                ncc = float((pa_ * pb_).sum() / (np.sqrt((pa_ ** 2).sum() * (pb_ ** 2).sum()) + 1e-9))
                if ncc < EDGE_NCC:
                    continue
                A.append(a); B.append(b)
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    A, B = np.array(A), np.array(B)
    key = np.round(A / EDGE_GRID).astype(int)
    _, first = np.unique(key[:, 0] * 100000 + key[:, 1], return_index=True)
    return A[first], B[first]


def match_all_pairs(imgs):
    """모든 쌍 매칭 → RANSAC → 판정식. 반환: {(i, j): (p, q, f)}  p, q 는 원래 이미지 좌표의 inlier"""
    n = len(imgs)
    centers = [np.array([im.shape[1] / 2.0, im.shape[0] / 2.0]) for im in imgs]
    feats = [detect(im) for im in imgs]
    pd = {}
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = match(feats[i], feats[j])
            H, m = bl_homography(pj - centers[j], pi - centers[i])
            if H is None or not enough_inliers(m.sum(), len(pi)):
                continue
            P, Q = pi[m].astype(np.float64), pj[m].astype(np.float64)
            if FEATURES == "sift_edge":                                   # 엣지 교점 대응점 추가
                A, Bq = edge_correspondences(imgs[i], imgs[j], P, Q)
                P, Q = np.vstack([P, A]), np.vstack([Q, Bq])
            pd[(i, j)] = (P, Q, focal_from_homography(H))
    return centers, pd


# ================================================================ 3. 원통 투영
def to_cyl(pts, c, f):
    """이미지 좌표 → 원통 좌표 (f*θ, f*h), 주점 = 이미지 중심"""
    x, y = pts[:, 0] - c[0], pts[:, 1] - c[1]
    return np.stack([f * np.arctan2(x, f), f * y / np.hypot(x, f)], axis=1)


def from_cyl(X, Y, c, f):
    """원통 좌표 → 이미지 좌표 (뒤쪽 반구는 -1e6)"""
    th = X / f
    bad = np.abs(th) >= (math.pi / 2 - 1e-3)
    th = np.where(bad, 0.0, th)
    x = f * np.tan(th) + c[0]
    y = Y / np.cos(th) + c[1]
    x = np.where(bad, -1e6, x)
    y = np.where(bad, -1e6, y)
    return x, y


# ================================================================ 4. 정합 (REW)
def _tps_kernel(a, b):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.where(d2 > 1e-20, d2 * np.log(np.maximum(d2, 1e-20)), 0.0)


class TPS:
    """thin-plate spline: 제어점 ctrl 에서 변위 disp 를 평활하게 보간 (lam: 평활 계수)"""

    def __init__(self, ctrl, disp, scale, lam=TPS_LAMBDA):
        self.scale = float(scale)
        self.c = ctrl / self.scale
        n = len(self.c)
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = _tps_kernel(self.c, self.c) + lam * np.eye(n)
        P = np.column_stack([np.ones(n), self.c])
        A[:n, n:] = P
        A[n:, :n] = P.T
        b = np.zeros((n + 3, 2))
        b[:n] = disp
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        self.w, self.a = sol[:n], sol[n:]

    def __call__(self, pts, chunk=4000):
        q = np.asarray(pts, np.float64) / self.scale
        out = np.empty((len(q), 2))
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            out[s:s + chunk] = _tps_kernel(qq, self.c) @ self.w + np.column_stack([np.ones(len(qq)), qq]) @ self.a
        return out


class REWEdge:
    """부모 이미지(원통 좌표 P) ← 자식 이미지(원통 좌표 Q) 의 REW 변환"""

    def __init__(self, P, Q, child_border_cyl, parent_border_cyl, width):
        G, _ = cv2.findHomography(Q, P, 0)          # 전역 homography (RANSAC inlier 전체로 최소제곱)
        self.G, self.Gi = G, np.linalg.inv(G)
        U = _pt(G, Q)
        D = U - P                                    # 부모 위치에서 본 남은 어긋남
        keep = np.ones(len(P), bool)
        rng = np.random.default_rng(0)
        scale = width
        tps = None
        for _ in range(5):                           # 변형으로 설명되지 않는 대응점 반복 제거
            idx = np.flatnonzero(keep)
            if len(idx) > TPS_MAX_CTRL:
                idx = np.sort(rng.choice(idx, TPS_MAX_CTRL, replace=False))
            tps = TPS(P[idx], D[idx], scale)
            r = np.linalg.norm(tps(P) - D, axis=1)
            mad = np.median(np.abs(r[keep] - np.median(r[keep])))
            thr = max(REW_OUT_MIN, REW_OUT_SIGMA * 1.4826 * mad)
            new = r < thr
            if (new == keep).all():
                break
            keep = new
        self.keep = keep
        self.n_ctrl = int(keep.sum())
        # 변위장을 격자 위에서 미리 계산 (부모 원통 좌표)
        box = np.vstack([_pt(G, child_border_cyl), parent_border_cyl])
        decay = REW_DECAY * width
        lo = box.min(0) - decay
        hi = box.max(0) + decay
        self.o = lo
        nx = int(math.ceil((hi[0] - lo[0]) / TPS_GRID)) + 2
        ny = int(math.ceil((hi[1] - lo[1]) / TPS_GRID)) + 2
        gx, gy = np.meshgrid(lo[0] + TPS_GRID * np.arange(nx), lo[1] + TPS_GRID * np.arange(ny))
        nodes = np.column_stack([gx.ravel(), gy.ravel()])
        disp = tps(nodes).reshape(ny, nx, 2)
        # η: 대응점 볼록 껍질 안 = 1, 밖으로 decay 만큼 멀어지면 0 (부드럽게)
        hull = cv2.convexHull(((P[keep] - lo) / TPS_GRID).astype(np.float32).reshape(-1, 1, 2))
        inside = np.zeros((ny, nx), np.uint8)
        cv2.fillConvexPoly(inside, np.round(hull).astype(np.int32), 1)
        dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5) * TPS_GRID
        t = np.clip(1 - dist / decay, 0, 1)
        eta = t * t * (3 - 2 * t)
        self.fx = (disp[..., 0] * eta).astype(np.float32)
        self.fy = (disp[..., 1] * eta).astype(np.float32)

    def disp(self, pts):
        n = len(pts)
        cols = 1024
        m = int(math.ceil(n / cols)) * cols         # remap 은 한 변이 32767 이하여야 해서 2차원으로 접음
        mx = np.zeros(m, np.float32)
        my = np.zeros(m, np.float32)
        mx[:n] = (pts[:, 0] - self.o[0]) / TPS_GRID
        my[:n] = (pts[:, 1] - self.o[1]) / TPS_GRID
        mx, my = mx.reshape(-1, cols), my.reshape(-1, cols)
        dx = cv2.remap(self.fx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()[:n]
        dy = cv2.remap(self.fy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()[:n]
        return np.stack([dx, dy], axis=1).astype(np.float64)

    def back(self, y):
        """부모 원통 좌표 → 자식 원통 좌표"""
        return _pt(self.Gi, y + self.disp(y))

    def fwd(self, q, iters=10):
        """자식 원통 좌표 → 부모 원통 좌표 (y = u - η g(y) 고정점 반복)"""
        u = _pt(self.G, q)
        y = u.copy()
        for _ in range(iters):
            y = u - self.disp(y)
        return y


def _border_points(w, h, n=40):
    xs = np.linspace(0, w - 1, n)
    ys = np.linspace(0, h - 1, n)
    return np.vstack([
        np.stack([xs, np.zeros(n)], 1), np.stack([xs, np.full(n, h - 1.0)], 1),
        np.stack([np.zeros(n), ys], 1), np.stack([np.full(n, w - 1.0), ys], 1),
    ]).astype(np.float64)


def align_rew(imgs, pairs=None):
    """매칭 → 초점거리 → 원통 좌표 → 최대 신장 트리를 따라 REW 로 기준 이미지에 연결."""
    n = len(imgs)
    centers, pd = pairs if pairs is not None else match_all_pairs(imgs)
    if not pd:
        raise RuntimeError("겹치는 쌍 없음")
    wmean = float(np.mean([im.shape[1] for im in imgs]))
    fs = [d[2] for d in pd.values() if d[2] and 0.3 * wmean < d[2] < 10 * wmean]
    f = float(np.median(fs)) if fs else 0.9 * wmean
    # 기준 = 대응점 수 합이 가장 큰 이미지, 대응점이 많은 쌍부터 연결 (최대 신장 트리)
    score = np.zeros(n)
    for (i, j), d in pd.items():
        score[i] += len(d[0])
        score[j] += len(d[0])
    ref = int(np.argmax(score))
    parent, order = {ref: None}, [ref]
    while True:
        best = None
        for (i, j), d in pd.items():
            if (i in parent) != (j in parent) and (best is None or len(d[0]) > len(best[2][0])):
                best = (i, j, d)
        if best is None:
            break
        i, j, d = best
        if i in parent:
            par, ch, Pp, Qc = i, j, d[0], d[1]
        else:
            par, ch, Pp, Qc = j, i, d[1], d[0]
        P = to_cyl(Pp, centers[par], f)
        Q = to_cyl(Qc, centers[ch], f)
        hc, wc = imgs[ch].shape[:2]
        hp, wp = imgs[par].shape[:2]
        edge = REWEdge(P, Q, to_cyl(_border_points(wc, hc), centers[ch], f),
                       to_cyl(_border_points(wp, hp), centers[par], f), wmean)
        parent[ch] = (par, edge)
        order.append(ch)
    comp = order                                    # 기준과 연결된 이미지 (연결 순서)

    def chain(k):
        path = []
        while parent[k] is not None:
            par, e = parent[k]
            path.append(e)
            k = par
        return path                                 # 자식 → ... → 기준 순서

    fwds, invs = [], []
    for k in comp:
        path, c = chain(k), centers[k]

        def fwd(pts, path=path, c=c):
            y = to_cyl(np.asarray(pts, np.float64), c, f)
            for e in path:
                y = e.fwd(y)
            return y

        def inv(X, Y, path=path, c=c):
            shp = X.shape
            y = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64)
            for e in reversed(path):
                y = e.back(y)
            x, yy = from_cyl(y[:, 0], y[:, 1], c, f)
            return x.reshape(shp), yy.reshape(shp)

        fwds.append(fwd)
        invs.append(inv)
    info = {"f": f, "ref": ref, "order": comp, "edges": {k: parent[k] for k in comp}}
    return [imgs[k] for k in comp], fwds, invs, info


# ================================================================ 5. 합성 (gain + 이음선 + LPAM)
def render(imgs, fwds, invs):
    """fwds[k](pts): 이미지 k 좌표 → 파노라마 좌표,  invs[k](X, Y): 파노라마 → 이미지 k 좌표"""
    boxes = []
    for im, fwd in zip(imgs, fwds):
        h, w = im.shape[:2]
        P = fwd(_border_points(w, h))
        P = P[np.all(np.isfinite(P), axis=1)]
        boxes.append((P.min(0), P.max(0)))
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    W0, H0 = hi - lo + 1
    s = min(1.0, MAX_PANO_SIDE / max(W0, H0), math.sqrt(MAX_PANO_PIXELS / (W0 * H0)))
    W, H = int(math.ceil(W0 * s)), int(math.ceil(H0 * s))
    tiles = []
    for im, inv, (bmin, bmax) in zip(imgs, invs, boxes):
        h, w = im.shape[:2]
        x0 = max(0, int(math.floor((bmin[0] - lo[0]) * s)) - 1)
        y0 = max(0, int(math.floor((bmin[1] - lo[1]) * s)) - 1)
        x1 = min(W, int(math.ceil((bmax[0] - lo[0]) * s)) + 2)
        y1 = min(H, int(math.ceil((bmax[1] - lo[1]) * s)) + 2)
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float64) / s + lo[0],
                           np.arange(y0, y1, dtype=np.float64) / s + lo[1])
        mx, my = inv(X, Y)
        mx = np.nan_to_num(mx, nan=-1e6, posinf=-1e6, neginf=-1e6).astype(np.float32)
        my = np.nan_to_num(my, nan=-1e6, posinf=-1e6, neginf=-1e6).astype(np.float32)
        warped = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
        tiles.append((x0, y0, warped, valid))
    return tiles, (W, H), (lo, s)


def gain_compensation(tiles, sigma_n=10.0, sigma_g=0.1):
    """Brown & Lowe (2007) 노출 보정 (baseline 과 동일)"""
    n = len(tiles)
    if n < 2:
        return tiles
    stats = []
    for i in range(n):
        xi, yi, wi, vi = tiles[i]
        for j in range(i + 1, n):
            xj, yj, wj, vj = tiles[j]
            ax0, ay0 = max(xi, xj), max(yi, yj)
            ax1 = min(xi + vi.shape[1], xj + vj.shape[1])
            ay1 = min(yi + vi.shape[0], yj + vj.shape[0])
            if ax1 - ax0 < 2 or ay1 - ay0 < 2:
                continue
            si = (slice(ay0 - yi, ay1 - yi, 2), slice(ax0 - xi, ax1 - xi, 2))
            sj = (slice(ay0 - yj, ay1 - yj, 2), slice(ax0 - xj, ax1 - xj, 2))
            m = vi[si] & vj[sj]
            N = int(m.sum())
            if N < 50:
                continue
            stats.append((i, j, N, wi[si][m].reshape(-1, 3).mean(0), wj[sj][m].reshape(-1, 3).mean(0)))
    if not stats:
        return tiles
    gains = np.ones((n, 3))
    for c in range(3):
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i, j, N, Ii, Ij in stats:
            for a, bb, Ia, Ib in ((i, j, Ii[c], Ij[c]), (j, i, Ij[c], Ii[c])):
                A[a, a] += N * (Ia * Ia / sigma_n ** 2 + 1.0 / sigma_g ** 2)
                A[a, bb] -= N * Ia * Ib / sigma_n ** 2
                b[a] += N / sigma_g ** 2
        for k in range(n):
            if A[k, k] == 0:
                A[k, k], b[k] = 1.0, 1.0
        try:
            gains[:, c] = np.clip(np.linalg.solve(A, b), 0.5, 2.0)
        except np.linalg.LinAlgError:
            pass
    out = []
    for (x0, y0, w, v), g in zip(tiles, gains):
        w2 = np.clip(w.astype(np.float32) * g.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
        out.append((x0, y0, w2, v))
    return out


def _seam_cost(A, B):
    """이음선 비용: 색 차이 + 기울기 차이 (값이 작은 곳으로 이음선이 지나감)"""
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = lambda g: np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return np.abs(a - b).mean(2) + 0.5 * np.abs(grad(ga) - grad(gb))


def _dp_seam(cost, start=None, end=None):
    """위 → 아래로 내려가는 최소 비용 경로 (행마다 열 하나, 이웃 행끼리 ±1 열)"""
    h, w = cost.shape
    big = 1e9
    acc = cost.astype(np.float64).copy()
    if start is not None:
        row = np.full(w, big)
        row[int(np.clip(start, 0, w - 1))] = acc[0, int(np.clip(start, 0, w - 1))]
        acc[0] = row
    back = np.zeros((h, w), np.int8)
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.r_[big, prev[:-1]]
        right = np.r_[prev[1:], big]
        stack = np.stack([left, prev, right])
        k = stack.argmin(0)
        acc[y] += stack[k, np.arange(w)]
        back[y] = k - 1
    if end is not None:
        x = int(np.clip(end, 0, w - 1))
    else:
        x = int(acc[-1].argmin())
    path = np.zeros(h, np.int32)
    for y in range(h - 1, -1, -1):
        path[y] = x
        x = int(np.clip(x + int(back[y, x]), 0, w - 1))
    return path


# ================================================================ [비교용 변형] 블렌딩 = baseline (노출 보정 + 멀티밴드)
def push_pull_fill(img, valid, levels=8):
    """유효하지 않은 영역을 주변 색으로 채움 (피라미드 경계에 검은색이 번지지 않게)"""
    imgs, ws = [img * valid[..., None]], [valid.astype(np.float32)]
    for _ in range(levels):
        if min(imgs[-1].shape[:2]) < 4:
            break
        imgs.append(cv2.pyrDown(imgs[-1]))
        ws.append(cv2.pyrDown(ws[-1]))
    filled = imgs[-1] / np.maximum(ws[-1], 1e-6)[..., None]
    for l in range(len(imgs) - 2, -1, -1):
        up = cv2.pyrUp(filled, dstsize=(imgs[l].shape[1], imgs[l].shape[0]))
        a = np.clip(ws[l], 0, 1)[..., None]
        cur = imgs[l] / np.maximum(ws[l], 1e-6)[..., None]
        filled = cur * a + up * (1 - a)
    return filled


def weight_image(img):
    """Brown & Lowe 가중치 W(x, y) = w(x)·w(y), w 는 사진 중심 1 → 가장자리 0 으로 선형 감소"""
    h, w = img.shape[:2]
    wx = 1 - np.abs(np.linspace(-1, 1, w))
    wy = 1 - np.abs(np.linspace(-1, 1, h))
    W = np.outer(wy, wx)
    return np.repeat((W * 254 + 1).astype(np.uint8)[..., None], 3, axis=2)


def stitch_q2_parts(imgs):
    sub, fwds, invs, info = align_rew(imgs)
    parts = compose_parts(sub, fwds, invs)
    parts["align"] = info
    return parts


# ---------------------------------------------------------------- Q1 평면 변환
def _plane_fwd(pts, c, f):
    return np.asarray(pts, np.float64)


def _plane_inv(X, Y, c, f):
    return X, Y


# ================================================================ 6. Q3·E1·E2: Brown & Lowe BA (회전 + 초점거리)
HUBER_PX = 2.0            # 논문: 최종 Huber 기준 2px
MAX_PTS_PER_PAIR = 300    # bundle adjustment 에 쓰는 쌍당 대응점 수


def bl_detect(img):
    """SIFT 만 (Brown & Lowe), 특징점 수 상한 없음"""
    gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = _resize_max(gray_, FEAT_MAX_SIDE)
    s = small.shape[1] / float(gray_.shape[1])
    kps, des = cv2.SIFT_create(nfeatures=0).detectAndCompute(small, None)   # nfeatures=0: 상한 없음
    if des is None or len(kps) < 4:
        return {}
    return {"SIFT": (np.float32([k.pt for k in kps]) / s, des, cv2.NORM_L2)}


def bl_enough_inliers(n_inl, n_match, relax=False):
    """Brown & Lowe (2007) 의 판정식: inlier > 8 + 0.3 * 매칭 수"""
    a = 0.15 if relax else 0.3
    return n_inl >= 8 + a * n_match


def _rays(pts, f):
    """이미지 좌표(중심 원점) → 카메라에서 나가는 단위 광선 (x, y, f)/|.|"""
    r = np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), float(f))]).astype(np.float64)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _procrustes(A, B):
    """R @ b ≈ a 가 되는 회전 R (Kabsch, SVD).  A, B : N x 3"""
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)])
    return U @ D @ Vt


def _component(n, links):
    """대응점 수 합이 가장 큰 이미지를 기준(ref)으로, 연결된 이미지 집합"""
    score = np.zeros(n)
    adj = {k: set() for k in range(n)}
    for lk in links:
        i, j, w = lk[0], lk[1], len(lk[2])
        score[i] += w
        score[j] += w
        adj[i].add(j)
        adj[j].add(i)
    ref = int(np.argmax(score))
    comp, stack = {ref}, [ref]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in comp:
                comp.add(v)
                stack.append(v)
    return ref, comp


def _subsample(*arrs):
    k = len(arrs[0])
    if k <= MAX_PTS_PER_PAIR:
        return arrs
    sel = np.linspace(0, k - 1, MAX_PTS_PER_PAIR).astype(int)
    return tuple(a[sel] for a in arrs)


def solve_rotations(n, ref, links):
    """전역 회전 추정 (간단한 bundle adjustment)"""
    R = {ref: np.eye(3)}
    while True:
        best = None
        for i, j, A, B, Rij in links:
            if (i in R) != (j in R) and (best is None or len(A) > len(best[2])):
                best = (i, j, A, B, Rij)
        if best is None:
            break
        i, j, _, _, Rij = best
        if i in R:
            R[j] = R[i] @ Rij      # R_i a = R_j b,  a = R_ij b
        else:
            R[i] = R[j] @ Rij.T
    for _ in range(30):
        for k in list(R):
            if k == ref:
                continue
            src, dst = [], []
            for i, j, A, B, _ in links:
                if i == k and j in R:
                    src.append(A)
                    dst.append(B @ R[j].T)
                elif j == k and i in R:
                    src.append(B)
                    dst.append(A @ R[i].T)
            if src:
                R[k] = _procrustes(np.vstack(dst), np.vstack(src))
    return R


def level_rotations(R):
    """수평 맞추기 (Brown & Lowe 2007, wave correction)"""
    ks = list(R)
    X = np.array([R[k][:, 0] for k in ks])
    ybar = np.mean([R[k][:, 1] for k in ks], axis=0)
    ybar /= np.linalg.norm(ybar) + 1e-12
    M = X.T @ X - 0.1 * len(ks) * np.outer(ybar, ybar)
    up = np.linalg.eigh(M)[1][:, 0]
    if up @ ybar < 0:
        up = -up
    z0 = np.mean([R[k][:, 2] for k in ks], axis=0)
    z0 -= (z0 @ up) * up
    if np.linalg.norm(z0) < 1e-6:
        return R
    z0 /= np.linalg.norm(z0)
    x0 = np.cross(up, z0)
    Rw = np.vstack([x0, up, z0])
    return {k: Rw @ R[k] for k in ks}


def rodrigues(v):
    R, _ = cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))
    return R


def bl_residuals(params, n, ref, links):
    """대응점 (p in i, q in j): q 를 카메라 j 광선 → 세계 → 카메라 i 로 투영했을 때 p 와의 차이 (px)"""
    R, f = {}, params[-n:]
    t = 0
    for k in range(n):
        if k == ref:
            R[k] = np.eye(3)
        else:
            R[k] = rodrigues(params[3 * t:3 * t + 3])
            t += 1
    res = []
    for i, j, P, Q in links:
        ray = np.column_stack([Q[:, 0], Q[:, 1], np.full(len(Q), f[j])])
        c = ray @ (R[i].T @ R[j]).T          # 카메라 i 좌표 = R_i^T R_j ray
        z = np.where(np.abs(c[:, 2]) < 1e-6, 1e-6, c[:, 2])
        proj = f[i] * c[:, :2] / z[:, None]
        res.append((proj - P).ravel())
    return np.concatenate(res), R, f


def bundle_adjust(n, ref, links, R0, f0, iters=60):
    """Levenberg–Marquardt + Huber (Brown & Lowe 4장)"""
    x = []
    for k in range(n):
        if k != ref:
            v, _ = cv2.Rodrigues(R0[ref].T @ R0[k])  # 기준 카메라 = 단위 회전
            x.extend(v.ravel())
    x = np.array(x + list(f0), np.float64)
    lam = 1e-3

    def cost_and_w(r, huber):
        a = np.abs(r)
        if huber is None:
            return float((r * r).sum()), np.ones_like(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        c = np.where(a <= huber, r * r, 2 * huber * a - huber * huber)
        return float(c.sum()), w

    for stage, huber in ((0, None), (1, HUBER_PX)):  # 처음엔 일반 최소제곱, 마지막은 Huber 2px
        for _ in range(iters):
            r, _, _ = bl_residuals(x, n, ref, links)
            cost, w = cost_and_w(r, huber)
            J = np.zeros((len(r), len(x)))
            for p in range(len(x)):
                h = 1e-6 if p < len(x) - n else 1e-3 * max(1.0, abs(x[p]))
                xp = x.copy()
                xp[p] += h
                J[:, p] = (bl_residuals(xp, n, ref, links)[0] - r) / h
            JW = J * w[:, None]
            A = JW.T @ J
            g = JW.T @ r
            improved = False
            for _ in range(10):
                try:
                    dx = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                xn = x + dx
                rn = bl_residuals(xn, n, ref, links)[0]
                cn, _ = cost_and_w(rn, huber)
                if cn < cost:
                    x, lam, improved = xn, max(lam / 3, 1e-9), True
                    break
                lam *= 10
            if not improved or np.linalg.norm(dx) < 1e-8 * (1 + np.linalg.norm(x)):
                break
    _, R, f = bl_residuals(x, n, ref, links)
    return R, f


# ================================================================ 7. Q3·E1·E2: BA 배치 위 REW 보정 (모든 겹침 쌍 중간점) + DIS 대응점


class REWEdgeBA(REWEdge):
    """BA 배치 위에서 남은 어긋남만 TPS 로 보정"""
    def __init__(self, P, Q, child_border_cyl, parent_border_cyl, width):
        G = np.eye(3)                                # 전역 모델 = BA (이미 전역 원통 좌표에 배치됨)
        self.G, self.Gi = G, np.linalg.inv(G)
        U = _pt(G, Q)
        D = U - P                                    # 부모 위치에서 본 남은 어긋남
        keep = np.ones(len(P), bool)
        rng = np.random.default_rng(0)
        scale = width
        tps = None
        for _ in range(5):                           # 변형으로 설명되지 않는 대응점 반복 제거
            idx = np.flatnonzero(keep)
            if len(idx) > TPS_MAX_CTRL:
                idx = np.sort(rng.choice(idx, TPS_MAX_CTRL, replace=False))
            tps = TPS(P[idx], D[idx], scale)
            r = np.linalg.norm(tps(P) - D, axis=1)
            mad = np.median(np.abs(r[keep] - np.median(r[keep])))
            thr = max(REW_OUT_MIN, REW_OUT_SIGMA * 1.4826 * mad)
            new = r < thr
            if (new == keep).all():
                break
            keep = new
        self.keep = keep
        self.n_ctrl = int(keep.sum())
        # 변위장을 격자 위에서 미리 계산 (부모 원통 좌표)
        box = np.vstack([_pt(G, child_border_cyl), parent_border_cyl])
        decay = REW_DECAY * width
        lo = box.min(0) - decay
        hi = box.max(0) + decay
        self.o = lo
        nx = int(math.ceil((hi[0] - lo[0]) / TPS_GRID)) + 2
        ny = int(math.ceil((hi[1] - lo[1]) / TPS_GRID)) + 2
        gx, gy = np.meshgrid(lo[0] + TPS_GRID * np.arange(nx), lo[1] + TPS_GRID * np.arange(ny))
        nodes = np.column_stack([gx.ravel(), gy.ravel()])
        disp = tps(nodes).reshape(ny, nx, 2)
        # η: 대응점 볼록 껍질 안 = 1, 밖으로 decay 만큼 멀어지면 0 (부드럽게)
        hull = cv2.convexHull(((P[keep] - lo) / TPS_GRID).astype(np.float32).reshape(-1, 1, 2))
        inside = np.zeros((ny, nx), np.uint8)
        cv2.fillConvexPoly(inside, np.round(hull).astype(np.int32), 1)
        dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5) * TPS_GRID
        t = np.clip(1 - dist / decay, 0, 1)
        eta = t * t * (3 - 2 * t)
        self.fx = (disp[..., 0] * eta).astype(np.float32)
        self.fy = (disp[..., 1] * eta).astype(np.float32)
PROJ = "cylindrical"        # E1 은 "spherical" 로 바꿔서 실행


def global_cyl(imgs, centers, pd):
    """baseline 의 BA 부분 (stitch_bl 과 동일). pd 형식만 모델 것 → baseline 것으로 변환"""
    n = len(imgs)
    bpd = {k: {"p": d[0] - centers[k[0]], "q": d[1] - centers[k[1]], "f": d[2]} for k, d in pd.items()}
    wmean = float(np.mean([im.shape[1] for im in imgs]))
    fs = [d["f"] for d in bpd.values() if d["f"] and 0.3 * wmean < d["f"]]
    f0 = float(np.median(fs)) if fs else 0.9 * wmean
    rlinks = []
    for (i, j), d in bpd.items():
        A, Bv = _subsample(_rays(d["p"], f0), _rays(d["q"], f0))
        rlinks.append((i, j, A, Bv, _procrustes(A, Bv)))
    ref, comp = _component(n, [(i, j, A) for i, j, A, _, _ in rlinks])
    comp = sorted(comp)
    idx = {k: t for t, k in enumerate(comp)}
    R0 = solve_rotations(len(comp), idx[ref], [(idx[i], idx[j], A, Bv, Rij) for i, j, A, Bv, Rij in rlinks if i in idx and j in idx])
    links = []
    for (i, j), d in bpd.items():
        if i in idx and j in idx:
            P, Q = _subsample(d["p"], d["q"])
            links.append((idx[i], idx[j], P, Q))
    R, f = bundle_adjust(len(comp), idx[ref], links, R0, [f0] * len(comp))
    R = level_rotations(dict(R))
    F = float(np.median(f))
    G = {}
    for k in comp:
        Rk, c, fk = R[idx[k]], centers[k], float(f[idx[k]])

        def fwd(pts, Rk=Rk, c=c, fk=fk):
            d = _rays(np.asarray(pts, np.float64) - c, fk) @ Rk.T
            if PROJ == "spherical":
                return np.stack([F * np.arctan2(d[:, 0], d[:, 2]), F * np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2]))], axis=1)
            return np.stack([F * np.arctan2(d[:, 0], d[:, 2]), F * d[:, 1] / np.hypot(d[:, 0], d[:, 2])], axis=1)

        def inv(X, Y, Rk=Rk, c=c, fk=fk):
            if PROJ == "spherical":
                th, ph = X / F, Y / F
                dx, dy, dz = np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)
            else:
                th, hh = X / F, Y / F
                dx, dy, dz = np.sin(th), hh, np.cos(th)
            cx = Rk[0, 0] * dx + Rk[1, 0] * dy + Rk[2, 0] * dz
            cy = Rk[0, 1] * dx + Rk[1, 1] * dy + Rk[2, 1] * dz
            cz = Rk[0, 2] * dx + Rk[1, 2] * dy + Rk[2, 2] * dz
            bad = cz < 1e-3
            cz = np.where(bad, 1.0, cz)
            x, y = fk * cx / cz + c[0], fk * cy / cz + c[1]
            x[bad] = -1e6
            y[bad] = -1e6
            return x, y
        G[k] = (fwd, inv)
    return comp, G, F


def dis_correspondences(imgs, G, pd, step=10, max_flow=30.0, grad_thr=8.0):
    """BA 로 배치한 원통 좌표에서 겹침 영역마다 DIS optical flow(밝기 기반) → 촘촘한 대응점."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    box = {}
    for k in G:
        h, w = imgs[k].shape[:2]
        P = G[k][0](_border_points(w, h))
        box[k] = (P.min(0), P.max(0))
    out = {}
    for (i, j) in pd:
        if i not in G or j not in G:
            continue
        lo = np.maximum(box[i][0], box[j][0])
        hi = np.minimum(box[i][1], box[j][1])
        if np.any(hi - lo < 40):
            continue
        X, Y = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        warped, valid = [], []
        for k in (i, j):
            mx, my = G[k][1](X, Y)
            mx = np.nan_to_num(mx, nan=-1e6).astype(np.float32)
            my = np.nan_to_num(my, nan=-1e6).astype(np.float32)
            h, w = gray[k].shape
            warped.append(cv2.remap(gray[k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
            valid.append((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1))
        ov = cv2.erode((valid[0] & valid[1]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if ov.sum() < 5000:
            continue
        f = dis.calc(warped[0], warped[1], None)
        b = dis.calc(warped[1], warped[0], None)
        hh, ww = ov.shape
        gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        bx = cv2.remap(b[..., 0], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
        by = cv2.remap(b[..., 1], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
        ok = ov & (np.hypot(f[..., 0] + bx, f[..., 1] + by) < 1.0) & (np.hypot(f[..., 0], f[..., 1]) < max_flow)
        g = warped[0].astype(np.float32)
        grad = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 2)
        ok &= grad > grad_thr
        sel = np.zeros_like(ok)
        sel[step // 2::step, step // 2::step] = True
        ys, xs = np.where(ok & sel)
        if len(xs) < 20:
            continue
        Xa, Ya = X[ys, xs], Y[ys, xs]
        pi = np.column_stack(G[i][1](Xa, Ya))
        pj = np.column_stack(G[j][1](Xa + f[ys, xs, 0], Ya + f[ys, xs, 1]))
        good = np.all(np.isfinite(pi), 1) & np.all(np.isfinite(pj), 1) & (pi[:, 0] > -1e5) & (pj[:, 0] > -1e5)
        out[(i, j)] = (pi[good], pj[good])
    return out


def align_ba_rew_mid(imgs, pairs, use_rew=True, dense=False, dense_cap=None):
    """모든 겹침 쌍의 대응점을 중간점으로 옮기는 TPS 를 사진마다 하나씩"""
    centers, pd = pairs
    comp, G, F = global_cyl(imgs, centers, pd)
    wmean = float(np.mean([im.shape[1] for im in imgs]))
    cs = set(comp)
    fwds, invs, edges = [], [], {}
    if not use_rew:
        return [imgs[k] for k in comp], [G[k][0] for k in comp], [G[k][1] for k in comp], {"F": F, "order": list(comp), "edges": {}, "pd": pd}
    pd_ba = pd
    dense_pd = {}
    if dense:                                         # REW 에만 촘촘한 대응점 추가 (BA 는 특징점 대응점만)
        dense_pd = dis_correspondences(imgs, G, pd)
        if dense_cap is not None:                     # 쌍마다 조밀 대응점 수 제한
            rng = np.random.default_rng(0)
            for k_ in list(dense_pd):
                a_, b_ = dense_pd[k_]
                cap = int(dense_cap * len(pd[k_][0])) if dense_cap < 50 else int(dense_cap)
                if len(a_) > cap:
                    idx = np.sort(rng.choice(len(a_), cap, replace=False))
                    dense_pd[k_] = (a_[idx], b_[idx])
        pd = {k: (np.vstack([d[0], dense_pd[k][0]]), np.vstack([d[1], dense_pd[k][1]]), d[2]) if k in dense_pd else d
              for k, d in pd.items()}
    for k in comp:
        own, tgt = [], []
        for (i, j), d in pd.items():
            if i not in cs or j not in cs or k not in (i, j):
                continue
            pk, po, o = (d[0], d[1], j) if k == i else (d[1], d[0], i)
            a, b = G[k][0](pk), G[o][0](po)
            own.append(a)
            tgt.append((a + b) / 2)
        gf, gi = G[k]
        if not own:
            fwds.append(gf)
            invs.append(gi)
            edges[k] = None
            continue
        own, tgt = np.vstack(own), np.vstack(tgt)
        h, w = imgs[k].shape[:2]
        bd = gf(_border_points(w, h))
        e = REWEdgeBA(tgt, own, bd, bd, wmean)
        edges[k] = e

        def fwd(pts, e=e, gf=gf):
            return e.fwd(gf(pts))

        def inv(X, Y, e=e, gi=gi):
            shp = X.shape
            y = e.back(np.column_stack([X.ravel(), Y.ravel()]).astype(np.float64))
            return gi(y[:, 0].reshape(shp), y[:, 1].reshape(shp))

        fwds.append(fwd)
        invs.append(inv)
    info = {"F": F, "order": list(comp), "edges": edges, "pd": pd_ba, "pd_rew": pd, "dense": dense_pd}
    return [imgs[k] for k in comp], fwds, invs, info


# ---------------------------------------------------------------- seam-cutting 경계 + 멀티밴드
def seam_labels_q(tiles, size, wtiles, cost_fn=None):
    """겹치는 두 사진마다 DP seam-cutting 으로 경계를 찾고 (색·기울기 차이가 작은 곳으로),"""
    W, H = size
    n = len(tiles)
    wins = np.zeros((n, H, W), np.int8)
    rect = [(x0, y0, x0 + v.shape[1], y0 + v.shape[0]) for x0, y0, _, v in tiles]
    cen = []
    for x0, y0, _, v in tiles:
        ys, xs = np.where(v)
        cen.append((xs.mean() + x0, ys.mean() + y0))
    for a in range(n):
        for b in range(a + 1, n):
            X0, Y0 = max(rect[a][0], rect[b][0]), max(rect[a][1], rect[b][1])
            X1, Y1 = min(rect[a][2], rect[b][2]), min(rect[a][3], rect[b][3])
            if X1 - X0 < 10 or Y1 - Y0 < 10:
                continue

            def crop(t):
                x0, y0, wimg, v = tiles[t]
                return wimg[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0], v[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            A, va = crop(a)
            Bc, vb = crop(b)
            O = va & vb
            if O.sum() < 500:
                continue
            ys, xs = np.where(O)
            r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            A, Bc, Oc = A[r0:r1, c0:c1], Bc[r0:r1, c0:c1], O[r0:r1, c0:c1]
            dx, dy = cen[b][0] - cen[a][0], cen[b][1] - cen[a][1]
            vertical = abs(dx) >= abs(dy)
            if not vertical:                           # 위아래로 붙으면 전치해서 같은 방식
                A, Bc, Oc = (np.ascontiguousarray(z.swapaxes(0, 1)) for z in (A, Bc, Oc))
            cost = (cost_fn or _seam_cost)(A, Bc)
            cost[~Oc] = 1e6
            seam = _dp_seam(cost)
            b_after = (dx if vertical else dy) > 0
            cols = np.arange(Oc.shape[1])[None, :]
            side_b = (cols > seam[:, None]) if b_after else (cols < seam[:, None])
            if not vertical:
                side_b, Oc = side_b.T, Oc.T
            Ya, Xa = Y0 + r0, X0 + c0
            wa = wins[a, Ya:Ya + Oc.shape[0], Xa:Xa + Oc.shape[1]]
            wb = wins[b, Ya:Ya + Oc.shape[0], Xa:Xa + Oc.shape[1]]
            wa += (Oc & ~side_b).astype(np.int8)
            wb += (Oc & side_b).astype(np.int8)
    best = np.full((H, W), -1, np.int32)
    label = np.full((H, W), -1, np.int32)
    for k, (x0, y0, _, v) in enumerate(tiles):
        h, w = v.shape
        sc = wins[k, y0:y0 + h, x0:x0 + w].astype(np.int32) * 1000 + wtiles[k][2][..., 0].astype(np.int32)
        sc[~v] = -1
        roi = best[y0:y0 + h, x0:x0 + w]
        better = sc > roi
        roi[better] = sc[better]
        label[y0:y0 + h, x0:x0 + w][better] = k
    return label


def multiband_label(tiles, size, label, bands=5):
    """multiband_blend 와 같고, 섞는 기준만 주어진 label (seam-cutting 결과)"""
    W, H = size
    union = label >= 0
    acc, wsum = [None] * bands, [None] * bands
    for k, (x0, y0, warped, v) in enumerate(tiles):
        h, w = v.shape
        full = np.zeros((H, W, 3), np.float32)
        val = np.zeros((H, W), bool)
        full[y0:y0 + h, x0:x0 + w] = warped.astype(np.float32)
        val[y0:y0 + h, x0:x0 + w] = v
        full = push_pull_fill(full, val)
        gi, gm = [full], [(label == k).astype(np.float32)]
        for _ in range(bands - 1):
            gi.append(cv2.pyrDown(gi[-1]))
            gm.append(cv2.pyrDown(gm[-1]))
        lp = [gi[l] - cv2.pyrUp(gi[l + 1], dstsize=(gi[l].shape[1], gi[l].shape[0])) for l in range(bands - 1)] + [gi[-1]]
        for l in range(bands):
            if acc[l] is None:
                acc[l] = np.zeros_like(lp[l])
                wsum[l] = np.zeros(gm[l].shape, np.float32)
            acc[l] += lp[l] * gm[l][..., None]
            wsum[l] += gm[l]
    out = acc[-1] / np.maximum(wsum[-1], 1e-6)[..., None]
    for l in range(bands - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(acc[l].shape[1], acc[l].shape[0])) + acc[l] / np.maximum(wsum[l], 1e-6)[..., None]
    pano = np.zeros((H, W, 3), np.uint8)
    pano[union] = np.clip(out[union] + 0.5, 0, 255).astype(np.uint8)
    return pano, union


# ================================================================ 8. Q3·E1·E2 매칭(국소 일관성) · E2 블렌딩
LC_K = 8                 # 국소 일관성: 이웃 수
LC_TAU = 3.0             # 국소 일관성: 이웃과 잔차가 이 안이면 같은 움직임 (px)
LC_MAX = 50.0            # 전역 H 잔차가 이보다 크면 버림 (px)


# ---------------------------------------------------------------- S2 매칭
def local_consistent(pa, pb, H, k=LC_K, tau=LC_TAU):
    """주변 k 개 짝과 잔차가 비슷하면 남김"""
    n = len(pa)
    if n <= k or H is None:
        return np.zeros(n, bool)
    r = cv2.perspectiveTransform(pb.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2) - pa
    d2 = ((pa[:, None, :] - pa[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nb = np.argpartition(d2, k, axis=1)[:, :k]
    agree = (np.linalg.norm(r[nb] - r[:, None, :], axis=2) < tau).sum(1)
    return (agree >= (k + 1) // 2) & (np.linalg.norm(r, axis=1) < LC_MAX)


def match_pairs_local(imgs, detect, local=False):
    """baseline 매칭 (ratio → RANSAC → 판정식). 판정식·초점거리는 전역 RANSAC inlier 로 (baseline 과 같음)."""
    n = len(imgs)
    centers = [np.array([im.shape[1] / 2.0, im.shape[0] / 2.0]) for im in imgs]
    feats = [detect(im) for im in imgs]
    pd, st = {}, {}
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = match(feats[i], feats[j])
            if len(pi) < 4:
                continue
            H, m = bl_homography(pj - centers[j], pi - centers[i])
            if H is None or not bl_enough_inliers(m.sum(), len(pi)):
                continue
            keep = m.copy()
            if local:
                T = np.array([[1, 0, centers[i][0]], [0, 1, centers[i][1]], [0, 0, 1]]) @ H @ \
                    np.array([[1, 0, -centers[j][0]], [0, 1, -centers[j][1]], [0, 0, 1]])
                keep = m | local_consistent(pi.astype(np.float64), pj.astype(np.float64), T)
            pd[(i, j)] = (pi[keep].astype(np.float64), pj[keep].astype(np.float64), focal_from_homography(H))
            st[(i, j)] = dict(cand=len(pi), ransac=int(m.sum()), kept=int(keep.sum()))
    return centers, pd, feats, st


# ---------------------------------------------------------------- S4 노출: gain + 비네팅
def radius_image(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - w / 2.0, yy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
    return np.repeat(r[..., None], 3, axis=2)


def exposure_vignetting(tiles, rtiles, step=4, sigma_g=0.1, iters=3):
    """log 영역에서 사진별·채널별 gain(g) 과 공통 비네팅 f(r) = α r² + β r⁴ 를 함께 추정."""
    n = len(tiles)
    # 흐린 밝기(σ=4)로 비교
    # 물체 경계(흐린 영상에서도 기울기가 큰 곳)와 사진 테두리 근처만 뺌
    blur, grads, vals = [], [], []
    for x0, y0, w, v in tiles:
        f = w.astype(np.float32)
        vb = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 4)
        fb = cv2.GaussianBlur(f * v[..., None], (0, 0), 4) / np.maximum(vb, 1e-3)[..., None]
        g = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        blur.append(fb)
        grads.append(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)))
        vals.append(cv2.erode(v.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    rows_i, rows_j, dr2, dr4, rhs = [], [], [], [], []
    for i in range(n):
        xi, yi, wi, vi = tiles[i]
        for j in range(i + 1, n):
            xj, yj, wj, vj = tiles[j]
            ax0, ay0 = max(xi, xj), max(yi, yj)
            ax1, ay1 = min(xi + vi.shape[1], xj + vj.shape[1]), min(yi + vi.shape[0], yj + vj.shape[0])
            if ax1 - ax0 < 8 or ay1 - ay0 < 8:
                continue
            si = (slice(ay0 - yi, ay1 - yi, step), slice(ax0 - xi, ax1 - xi, step))
            sj = (slice(ay0 - yj, ay1 - yj, step), slice(ax0 - xj, ax1 - xj, step))
            Ii, Ij = blur[i][si], blur[j][sj]
            m = vals[i][si] & vals[j][sj] & (grads[i][si] < 20) & (grads[j][sj] < 20)
            m &= (Ii.min(2) > 8) & (Ij.min(2) > 8) & (Ii.max(2) < 245) & (Ij.max(2) < 245)
            if m.sum() < 200:
                continue
            ri, rj = rtiles[i][2][si][..., 0][m], rtiles[j][2][sj][..., 0][m]
            li, lj = np.log(Ii[m]), np.log(Ij[m])                  # (N, 3)
            rows_i.append(np.full(len(ri), i))
            rows_j.append(np.full(len(ri), j))
            dr2.append(ri ** 2 - rj ** 2)
            dr4.append(ri ** 4 - rj ** 4)
            rhs.append(lj - li)
    if not rhs:
        return tiles, None
    ii, jj = np.concatenate(rows_i), np.concatenate(rows_j)
    d2, d4, Y = np.concatenate(dr2), np.concatenate(dr4), np.concatenate(rhs)
    N = len(ii)
    nu = 3 * n + 2                                                     # [log g (n×3), α, β]
    sol = np.zeros(nu)
    w = np.ones((N, 3))
    for _ in range(iters):
        A = np.zeros((nu, nu))
        b = np.zeros(nu)
        for c in range(3):
            # 한 식: lg[i,c] - lg[j,c] + α d2 + β d4 = Y[:, c]
            idx_i, idx_j = 3 * ii + c, 3 * jj + c
            wc = w[:, c]
            cols = [(idx_i, 1.0), (idx_j, -1.0)]
            for (ca, sa) in cols:
                for (cb, sb) in cols:
                    np.add.at(A, (ca, cb), wc * sa * sb)
                np.add.at(A, (ca, 3 * n), wc * sa * d2)
                np.add.at(A, (ca, 3 * n + 1), wc * sa * d4)
                np.add.at(A, (3 * n, ca), wc * sa * d2)
                np.add.at(A, (3 * n + 1, ca), wc * sa * d4)
                np.add.at(b, ca, wc * sa * Y[:, c])
            A[3 * n, 3 * n] += (wc * d2 * d2).sum()
            A[3 * n, 3 * n + 1] += (wc * d2 * d4).sum()
            A[3 * n + 1, 3 * n] += (wc * d2 * d4).sum()
            A[3 * n + 1, 3 * n + 1] += (wc * d4 * d4).sum()
            b[3 * n] += (wc * d2 * Y[:, c]).sum()
            b[3 * n + 1] += (wc * d4 * Y[:, c]).sum()
        lam = N / (sigma_g ** 2) * 1e-4                                # gain 을 1(log 0) 쪽으로 약하게
        A[np.arange(3 * n), np.arange(3 * n)] += lam
        sol = np.linalg.solve(A + 1e-9 * np.eye(nu), b)
        lg, al, be = sol[:3 * n].reshape(n, 3), sol[3 * n], sol[3 * n + 1]
        res = Y - (lg[ii] - lg[jj] + (al * d2 + be * d4)[:, None])
        s = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = np.minimum(1.0, 1.5 * s / (np.abs(res) + 1e-9))           # Huber
    lg, al, be = sol[:3 * n].reshape(n, 3), sol[3 * n], sol[3 * n + 1]
    # 보정 배율의 평균이 1 이 되게 맞춤 (전체 밝기 유지)
    num, den = np.zeros(3), 0.0
    for (x0, y0, wimg, v), (_, _, rimg, _), g in zip(tiles, rtiles, lg):
        r = rimg[..., 0][v]
        num += np.exp(al * r * r + be * r ** 4).sum() * np.exp(g)
        den += len(r)
    lg -= np.log(num / max(den, 1.0))[None, :]
    out = []
    for (x0, y0, wimg, v), (_, _, rimg, _), g in zip(tiles, rtiles, lg):
        r = rimg[..., 0]
        fac = np.exp(al * r * r + be * r ** 4)[..., None] * np.exp(g)[None, None, :]
        out.append((x0, y0, np.clip(wimg.astype(np.float32) * fac + 0.5, 0, 255).astype(np.uint8), v))
    return out, dict(gains=np.exp(lg), alpha=float(al), beta=float(be), corner=float(np.exp(al + be)))


# ---------------------------------------------------------------- S5 경계: 어긋남·물체 경계를 피하는 seam
SEAM_W_FLOW = 10.0       # 어긋남 1px 당 비용 (색 차이 10 과 같게)
SEAM_W_EDGE = 0.25       # 물체 경계(기울기) 비용


def _flow_mag(A, Bc, O):
    ga, gb = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Bc, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f, b = dis.calc(ga, gb, None), dis.calc(gb, ga, None)
    hh, ww = ga.shape
    gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    bx = cv2.remap(b[..., 0], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
    by = cv2.remap(b[..., 1], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
    mag = np.hypot(f[..., 0], f[..., 1])
    bad = np.hypot(f[..., 0] + bx, f[..., 1] + by) > 1.0             # 양방향 불일치 = 믿을 수 없음 → 어긋남으로 봄
    mag = np.where(bad, np.maximum(mag, 3.0), mag)
    return np.minimum(mag, 10.0), ga.astype(np.float32), gb.astype(np.float32)


_BASE_SEAM_COST = _seam_cost                     # 색·기울기 차이 (Q2·⑦ 과 같음)


def seam_cost_aware(A, Bc, O=None):
    flow, ga, gb = _flow_mag(A, Bc, O)
    edge = 0.5 * (cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1)) +
                  cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1)))
    return _BASE_SEAM_COST(A, Bc) + SEAM_W_FLOW * cv2.GaussianBlur(flow, (0, 0), 2) + SEAM_W_EDGE * edge


def seam_labels_s(tiles, size, wtiles):
    """seam_labels_q 와 같고 비용에 어긋남(DIS flow)·물체 경계를 더함"""
    return seam_labels_q(tiles, size, wtiles, cost_fn=seam_cost_aware)


def maxweight_labels(tiles, size, wtiles):
    W, H = size
    dmax = np.full((H, W), -1.0, np.float32)
    label = np.full((H, W), -1, np.int32)
    for k, (x0, y0, _, v) in enumerate(tiles):
        h, w = v.shape
        d = wtiles[k][2][..., 0].astype(np.float32)
        d[~v] = -1
        roi = dmax[y0:y0 + h, x0:x0 + w]
        better = d > roi
        roi[better] = d[better]
        label[y0:y0 + h, x0:x0 + w][better] = k
    return label


# ---------------------------------------------------------------- E2 전체
def stitch_e2_parts(imgs, stage=5):
    """E2 단계: 3 = gain, 4 = +비네팅, 5 = +어긋남 회피 seam"""
    centers, pd, _, _ = match_pairs_local(imgs, bl_detect, local=True)
    sub, fwds, invs, info = align_ba_rew_mid(imgs, (centers, pd), True, dense=True, dense_cap=1.0)
    tiles, size, _ = render(sub, fwds, invs)
    wtiles, _, _ = render([weight_image(im) for im in sub], fwds, invs)
    if stage >= 4:
        rtiles, _, _ = render([radius_image(im) for im in sub], fwds, invs)
        gained, _ = exposure_vignetting(tiles, rtiles)
    else:
        gained = gain_compensation(tiles)
    label = seam_labels_s(gained, size, wtiles) if stage >= 5 else maxweight_labels(gained, size, wtiles)
    pano, mask = multiband_label(gained, size, label)
    ys, xs = np.where(mask)
    return {"out": pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy(), "imgs": sub, "fwds": fwds, "invs": invs,
            "tiles": gained, "label": label, "align": info}


# ================================================================ 10. baseline (팀원 IS_ws 의 B0, 이름 앞에 b0)
_B0_LAST = {}            # 평가용 캡처
# 알고리즘 설정: 비교 실험에서는 동일한 값으로 고정하세요.
B0_WORK_MEGAPIX = 1.0        # 특징점 처리용 영상의 최대 면적; 출력은 원본 축척
B0_SIFT_NFEATURES = 6000
B0_Q1_RATIO = 0.75
B0_Q1_RANSAC_PX = 3.0       # 특징점을 계산한 축소 영상 좌표 기준
B0_MIN_INLIERS = 10
# Keep highly overlapping required inputs connected. The Brown–Lowe confidence
# n_inliers/(8 + 0.3*n_matches) is below 10/3, so 3.5 disables the native
# "too-close image" suppression. The graph's LOWER acceptance threshold stays 1.
B0_PAIR_CLOSE_CONFIDENCE_LIMIT = 3.5
B0_Q2_MATCH_CONF = 0.65     # OpenCV detail matcher 설정; Q1_RATIO와 의미가 다름
B0_Q2_GRAPH_CONF = 1.0      # 연결 그래프 및 bundle adjustment의 confidence 기준
B0_Q2_WORK_MEGAPIX = 1.0   # Q1과 독립적으로 특징점 처리 해상도를 실험
B0_Q2_BLEND_BANDS = 5      # 넓은 밝기 변화와 세부 윤곽을 다른 해상도에서 합성
B0_Q3_WORK_MEGAPIX = 1.0
B0_Q3_MATCH_CONF = 0.45
B0_Q3_GRAPH_CONF = 1.0
B0_MAX_CANVAS_PIXELS = 24_000_000
B0_MAX_SIDE = 30_000        # OpenCV remap의 크기 제한 및 과도한 메모리 사용 방지
B0_VERBOSE = False         # True: 팀원 코드의 진행 로그 출력


def _b0_log(message):
    if B0_VERBOSE:
        print(message)


class _b0_PlanarProjectionError(ValueError):
    """평면 지평선을 넘은 경우 이미 추정한 카메라를 원통/구면 합성에서 재사용."""
    def __init__(self, message, images, intrinsics, rotations, scale, alignment=None):
        super().__init__(message)
        self.images = images
        self.intrinsics = intrinsics
        self.rotations = rotations
        self.scale = scale
        self.alignment = alignment


def _b0_read_images(img_paths, expected):
    if len(img_paths) != expected:
        raise ValueError(f"Expected {expected} image paths, got {len(img_paths)}.")
    images = []
    for path in img_paths:
        image = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_COLOR)   # 평가용: 배열도 받음
        if image is None or image.size == 0:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        if min(h, w) < 16 or max(h, w) > B0_MAX_SIDE:
            raise ValueError(f"Unsupported image size {w}x{h}: {path}")
        images.append(image)
    return images


def _b0_working_images(images, work_megapix=B0_WORK_MEGAPIX):
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


def _b0_check_canvas(width, height):
    width, height = int(width), int(height)
    if (width <= 0 or height <= 0 or max(width, height) > B0_MAX_SIDE
            or width * height > B0_MAX_CANVAS_PIXELS):
        raise ValueError(
            f"Unsafe panorama size: {width}x{height}. "
            "Check matches / camera estimates, or resize the input images."
        )


def _b0_union_roi(corners, sizes):
    left = min(int(c[0]) for c in corners)
    top = min(int(c[1]) for c in corners)
    right = max(int(c[0]) + int(s[0]) for c, s in zip(corners, sizes))
    bottom = max(int(c[1]) + int(s[1]) for c, s in zip(corners, sizes))
    _b0_check_canvas(right - left, bottom - top)
    return left, top, right - left, bottom - top


def _b0_image_corners(image):
    h, w = image.shape[:2]
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])


def _b0_multiband_blend(warped, masks, corners, *, label="Q2"):
    """이미지, 유효 마스크, 평면 원점을 함께 반환하여 재투영 좌표계를 보존."""
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    left, top, width, height = _b0_union_roi(corners, sizes)
    blender = cv2.detail_MultiBandBlender()
    # 작은 입력에서도 피라미드 깊이가 영상 크기를 넘지 않도록 제한.
    bands = min(B0_Q2_BLEND_BANDS, max(1, int(np.floor(np.log2(min(width, height))))))
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
    _b0_log(f"[{label}] blend: multi-band, bands={bands}")
    return (np.ascontiguousarray(result[roi]),
            valid[roi].astype(np.float32), origin)


def _b0_hard_composite(warped, masks, corners, *, label):
    """유효 마스크의 픽셀을 그대로 선택. 겹치는 영역은 뒤 입력이 우선."""
    _B0_LAST["layers"] = (list(warped), list(masks), list(corners))
    sizes = [(im.shape[1], im.shape[0]) for im in warped]
    left, top, width, height = _b0_union_roi(corners, sizes)
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
    _b0_log(f"[{label}] hard composition: later input wins; no pixel blending")
    return np.ascontiguousarray(result[roi]), coverage[roi].astype(np.float32), origin


def _b0_maximum_weight_masks(masks, corners):
    """경계 거리 최대인 영상에 소유권을 부여; graph-cut 없이 초기 blend mask 생성."""
    sizes = [(mask.shape[1], mask.shape[0]) for mask in masks]
    left, top, width, height = _b0_union_roi(corners, sizes)
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


def _b0_brown_lowe_compose(warped, masks, corners, *, label):
    """과제 프로필: Q1/Q2/Q3/E1은 raw, E2는 global gain + multi-band."""
    if label != "E2":
        _b0_log(f"[{label}] Brown-Lowe assignment baseline: all postprocessing OFF")
        return _b0_hard_composite(warped, masks, corners, label=label)
    compensator = cv2.detail_GainCompensator()
    compensator.feed(corners, warped, masks)
    corrected = [compensator.apply(i, corner, image.copy(), mask)
                 for i, (image, mask, corner) in enumerate(zip(warped, masks, corners))]
    selected = _b0_maximum_weight_masks(masks, corners)
    _B0_LAST["e2"] = (corrected, selected, list(masks), list(corners))
    _b0_log(f"[{label}] Brown-Lowe assignment baseline: global gain + max-weight masks "
         "+ multi-band; no block gain or graph-cut")
    return _b0_multiband_blend(corrected, selected, corners, label=label)


def _b0_compose_planar(warped, masks, corners, *, label, advanced=True, seam_megapix=None):
    """과제 baseline 프로필: Q1~E1 덮어쓰기, E2 gain + 가중치 최대 + 멀티밴드"""
    return _b0_brown_lowe_compose(warped, masks, corners, label=label)


def b0_q1(img_paths):
    """Q1: 두 영상을 첫 번째 영상의 평면 좌표에 정합. BGR uint8 반환."""
    cv2.setRNGSeed(0)
    images = _b0_read_images(img_paths, 2)
    work, scales = _b0_working_images(images)
    sift = cv2.SIFT_create(nfeatures=B0_SIFT_NFEATURES)
    keypoints, descriptors = [], []
    for image in work:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kp) < B0_MIN_INLIERS:
            raise ValueError("Q1: too few features; use overlapping textured images.")
        keypoints.append(kp)
        descriptors.append(desc)

    # query = image 2, train = image 1: H21의 방향을 일관되게 유지.
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors[1], descriptors[0], k=2)
    good = [pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < B0_Q1_RATIO * pair[1].distance]
    if len(good) < B0_MIN_INLIERS:
        raise ValueError(f"Q1: only {len(good)} matches passed the ratio test.")
    src = np.float32([keypoints[1][m.queryIdx].pt for m in good])
    dst = np.float32([keypoints[0][m.trainIdx].pt for m in good])
    h_work, inliers = cv2.findHomography(
        src, dst, cv2.RANSAC, B0_Q1_RANSAC_PX, maxIters=5000, confidence=0.995
    )
    if h_work is None or inliers is None or int(inliers.sum()) < B0_MIN_INLIERS:
        raise ValueError("Q1: reliable homography could not be estimated.")
    # 축소 영상에서 구한 H를 원본 영상의 좌표로 복원.
    h21 = np.linalg.inv(scales[0]) @ h_work @ scales[1]
    if not np.isfinite(h21).all() or np.linalg.matrix_rank(h21) < 3:
        raise ValueError("Q1: invalid homography.")
    _B0_LAST["h21"] = h21
    corners2 = _b0_image_corners(images[1])
    homogeneous = np.column_stack([corners2, np.ones(4)]) @ h21.T
    denom = homogeneous[:, 2]
    # 사영의 무한대 선이 영상 내부를 통과하면 코너만으로 안전한 ROI를 구할 수 없음.
    if np.min(np.abs(denom)) < 1e-8 or not (np.all(denom > 0) or np.all(denom < 0)):
        raise ValueError("Q1: homography crosses the projective horizon.")
    projected = homogeneous[:, :2] / denom[:, None]
    all_corners = np.vstack([_b0_image_corners(images[0]), projected])
    low = np.floor(all_corners.min(axis=0))
    high = np.ceil(all_corners.max(axis=0))
    width, height = map(int, high - low)
    _b0_check_canvas(width, height)
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
    _b0_log(f"[Q1] keypoints={[len(k) for k in keypoints]}, "
         f"matches={len(good)}, inliers={int(inliers.sum())}/{len(good)}")
    corners = [(0, 0), (0, 0)]
    result, _, _ = _b0_brown_lowe_compose(warped, masks, corners, label="Q1")
    _b0_log(f"[Q1] output={result.shape[1]}x{result.shape[0]} (planar)")
    return result


def _b0_require_connected_graph(pairwise, count, *, label="Q2", confidence=B0_Q2_GRAPH_CONF):
    """전체 입력이 BA에 사용할 연결 그래프에 포함되는지 확인."""
    adjacency = [set() for _ in range(count)]
    for match in pairwise:
        i, j = int(match.src_img_idx), int(match.dst_img_idx)
        if not (0 <= i < j < count):
            continue
        _b0_log(f"[{label}] pair {i+1}-{j+1}: inliers={match.num_inliers}, "
             f"confidence={match.confidence:.3f}")
        if match.num_inliers >= B0_MIN_INLIERS and match.confidence > confidence:
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


def _b0_cylindrical_panorama(planar, valid, origin, scale, *, label="Q2"):
    """합성이 끝난 평면 파노라마 전체를 원통 좌표로 한 번 재투영."""
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
    _b0_check_canvas(u1 - u0 + 1, v1 - v0 + 1)

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
    _b0_log(f"[{label}] cylindrical canvas={u1-u0+1}x{v1-v0+1}, "
         "no crop (full reprojected image)")
    return result


def _b0_make_pairwise_matcher(match_conf):
    """Configure close-pair suppression explicitly, including older bindings."""
    try:
        return (cv2.detail_BestOf2NearestMatcher(
            False, match_conf, 6, 6, B0_PAIR_CLOSE_CONFIDENCE_LIMIT), False)
    except TypeError:
        # OpenCV 4.5.4 exposes only four arguments and hard-codes a limit of 3.
        return cv2.detail_BestOf2NearestMatcher(False, match_conf, 6, 6), True


def _b0_restore_close_pair_confidence(pairwise, *, label):
    """Undo only the old binding's high-confidence suppression, not weak matches."""
    restored = []
    for match in pairwise:
        if match.src_img_idx < 0 or match.dst_img_idx < 0 or match.src_img_idx == match.dst_img_idx:
            continue
        matches = match.getMatches()
        if not matches or not B0_MIN_INLIERS <= match.num_inliers <= len(matches):
            continue
        confidence = match.num_inliers / (8.0 + 0.3 * len(matches))
        matrix = match.H
        if (match.confidence == 0 and 3.0 < confidence <= B0_PAIR_CLOSE_CONFIDENCE_LIMIT
                and matrix is not None and matrix.shape == (3, 3)
                and np.isfinite(matrix).all() and np.linalg.matrix_rank(matrix) == 3):
            match.confidence = confidence
            restored.append([int(match.src_img_idx), int(match.dst_img_idx)])
    if restored:
        _b0_log(f"[{label}] restored {len(restored)} directed high-confidence matches "
             "suppressed by the legacy close-pair rule")
    return restored


def _b0_camera_planar_mosaic(img_paths, *, expected, label, work_megapix,
                          match_conf, graph_conf, grid_cols=None, advanced=True):
    """모든 입력의 카메라 관계를 추정하고 공통 평면 모자이크 한 장을 반환."""
    cv2.setRNGSeed(0)
    images = _b0_read_images(img_paths, expected)
    work, scales = _b0_working_images(images, work_megapix)
    sift = cv2.SIFT_create(nfeatures=B0_SIFT_NFEATURES)
    features = []
    for index, image in enumerate(work):
        feature = cv2.detail.computeImageFeatures2(sift, image)
        feature.img_idx = index
        if len(feature.getKeypoints()) < B0_MIN_INLIERS:
            raise ValueError(f"{label}: image {index+1} has too few features.")
        if len(feature.getKeypoints()) < B0_MIN_INLIERS:
            raise ValueError(f"{label}: image {index+1} has too few SIFT features.")
        features.append(feature)
    _b0_log(f"[{label}] keypoints={[len(f.getKeypoints()) for f in features]}")

    matcher, legacy_close_pair_rule = _b0_make_pairwise_matcher(match_conf)
    pairwise = (matcher.apply2(features) if grid_cols is None else
                matcher.apply2(features, _b0_grid_pair_mask(expected, grid_cols)))
    matcher.collectGarbage()
    if legacy_close_pair_rule:
        _b0_restore_close_pair_confidence(pairwise, label=label)
    native_pairwise = pairwise
    # leaveBiggestComponent를 호출하지 않음: 입력 일부 누락을 성공으로 처리하지 않음.
    _b0_require_connected_graph(pairwise, len(images), label=label, confidence=graph_conf)

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
    ok, cameras = adjuster.apply(features, pairwise, cameras)
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
    _b0_log(f"[{label}] focal_px={[round(float(k[0, 0]), 2) for k in intrinsics]}")
    # 모든 입력에 같은 평면 투영기와 광축을 사용한다.
    plane_warper = cv2.PyRotationWarper("plane", warp_scale)
    corners, sizes = [], []
    for image, k, rotation in zip(images, intrinsics, rotations):
        # 평면의 무한대 선을 가로지르거나 카메라 뒤쪽에 있는 입력은 거부.
        # z는 입력 좌표의 선형 함수이므로 네 코너의 검사로 전체를 판별 가능.
        points = np.column_stack([_b0_image_corners(image), np.ones(4)])
        rays = points @ (rotation @ np.linalg.inv(k)).T
        if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 1e-6):
            raise _b0_PlanarProjectionError(
                f"{label}: input crosses the planar projection horizon. "
                "A single planar panorama requires a field of view below 180 degrees.",
                images, intrinsics, rotations, warp_scale,
                alignment=(features,pairwise,scales,graph_conf)
            )
        roi = plane_warper.warpRoi((image.shape[1], image.shape[0]), k, rotation)
        _b0_check_canvas(roi[2], roi[3])
        corners.append(roi[:2])
        sizes.append(roi[2:])
    _b0_union_roi(corners, sizes)  # 큰 배열을 할당하기 전에 전체 출력 크기 검증
    _B0_LAST["warp"] = (plane_warper, intrinsics, rotations)

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
    planar, valid, origin = _b0_compose_planar(
        warped, masks, actual_corners, label=label, advanced=advanced
    )
    _b0_log(f"[{label}] planar mosaic={planar.shape[1]}x{planar.shape[0]} ({expected}/{expected} images)")
    return planar, valid, origin, warp_scale


def _b0_q2_planar_mosaic(img_paths):
    """Q2: 세 입력을 공통 평면에서 완전히 합성한 뒤 반환."""
    return _b0_camera_planar_mosaic(
        img_paths, expected=3, label="Q2", work_megapix=B0_Q2_WORK_MEGAPIX,
        match_conf=B0_Q2_MATCH_CONF, graph_conf=B0_Q2_GRAPH_CONF
    )


def b0_q2(img_paths):
    """3장 -> 평면 파노라마 1장 -> 하나의 원통 좌표계. BGR uint8 반환."""
    planar, valid, origin, warp_scale = _b0_q2_planar_mosaic(img_paths)
    _b0_log("[Q2] reprojecting ONE completed planar mosaic to ONE cylinder")
    result = _b0_cylindrical_panorama(planar, valid, origin, warp_scale)
    _b0_log(f"[Q2] output={result.shape[1]}x{result.shape[0]} "
         "(planar stitching -> cylindrical reprojection, 3/3 images)")
    return result


def _b0_grid_pair_mask(count, columns):
    """Raster 순서 격자의 가로/세로/대각 이웃만 매칭. 행 끝과 다음 행 시작은 제외."""
    mask = np.zeros((count, count), dtype=np.uint8)
    for i in range(count):
        for j in range(i + 1, count):
            if max(abs(i // columns - j // columns),
                   abs(i % columns - j % columns)) <= 1:
                mask[i, j] = mask[j, i] = 1
    return mask


def _b0_q3_planar_mosaic(img_paths, *, label="Q3", advanced=True):
    """2x4 입력의 모든 영상을 포함한 단일 평면 모자이크."""
    cv2.setRNGSeed(0)
    images = _b0_read_images(img_paths, 8)
    result = _b0_camera_planar_mosaic(
        images, expected=8, label=label, work_megapix=B0_Q3_WORK_MEGAPIX,
        match_conf=B0_Q3_MATCH_CONF, graph_conf=B0_Q3_GRAPH_CONF, grid_cols=4,
        advanced=advanced
    )
    planar, _, _, _ = result
    _b0_log(f"[{label}] planar mosaic={planar.shape[1]}x{planar.shape[0]} (8/8 images)")
    return result


def _b0_wide_surface_mosaic(geometry, *, projection, label, advanced=True):
    """단일 평면에 담을 수 없는 8장을 하나의 공통 원통/구면 좌표계에서 합성."""
    if projection not in ("cylindrical", "spherical"):
        raise ValueError(f"Unsupported wide projection: {projection}")
    warper = cv2.PyRotationWarper(projection, geometry.scale)
    corners, sizes = [], []
    for image, k, rotation in zip(geometry.images, geometry.intrinsics, geometry.rotations):
        roi = warper.warpRoi((image.shape[1], image.shape[0]), k, rotation)
        _b0_check_canvas(roi[2], roi[3])
        corners.append(roi[:2])
        sizes.append(roi[2:])
    _b0_union_roi(corners, sizes)
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
    _B0_LAST["warp"] = (warper, geometry.intrinsics, geometry.rotations)
    result, _, _ = _b0_compose_planar(warped, masks, actual_corners, label=label, advanced=advanced)
    _b0_log(f"[{label}] output={result.shape[1]}x{result.shape[0]} "
         f"(shared {projection} coordinates, 8/8 images, no planar reprojection)")
    return result


def _b0_wide_cylindrical_mosaic(geometry):
    """넓은 시야의 Q3 입력을 공통 원통에서 합성."""
    return _b0_wide_surface_mosaic(geometry, projection="cylindrical", label="Q3")


def b0_q3(img_paths):
    """8장 원통 정합. 평면 합성 가능 시 기존 경로, 지평선을 넘으면 공통 원통 합성."""
    try:
        planar, valid, origin, scale = _b0_q3_planar_mosaic(img_paths)
    except _b0_PlanarProjectionError as geometry:
        _b0_log("[Q3] field of view exceeds the common plane; switching to ONE shared cylinder")
        return _b0_wide_cylindrical_mosaic(geometry)
    _b0_log("[Q3] reprojecting ONE completed planar mosaic to ONE cylinder")
    return _b0_cylindrical_panorama(planar, valid, origin, scale, label="Q3")


def _b0_spherical_panorama(planar, valid, origin, scale, *, label):
    """완성된 평면 파노라마 한 장을 공통 구면의 경도/위도 좌표로 변환."""
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
    _b0_check_canvas(u1 - u0 + 1, v1 - v0 + 1)
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
    _b0_log(f"[{label}] spherical canvas={u1-u0+1}x{v1-v0+1}, "
         "no crop (full reprojected image)")
    return result


def _b0_spherical_stitch(img_paths, *, label, advanced):
    """8장 구면 정합. 좁은 시야는 평면 재투영, 넓은 시야는 공통 구면 합성."""
    try:
        planar, valid, origin, scale = _b0_q3_planar_mosaic(
            img_paths, label=label, advanced=advanced
        )
    except _b0_PlanarProjectionError as geometry:
        _b0_log(f"[{label}] field of view exceeds the common plane; switching to ONE shared sphere")
        return _b0_wide_surface_mosaic(
            geometry, projection="spherical", label=label, advanced=advanced
        )
    _b0_log(f"[{label}] reprojecting ONE completed planar mosaic to ONE sphere")
    return _b0_spherical_panorama(planar, valid, origin, scale, label=label)


def b0_e1(img_paths):
    """8장의 공통 구면 정합 + 기본 feather 블렌딩."""
    return _b0_spherical_stitch(img_paths, label="E1", advanced=False)


def b0_e2(img_paths):
    """8장의 공통 구면 정합 + 노출 보정 + multi-band 블렌딩."""
    return _b0_spherical_stitch(img_paths, label="E2", advanced=True)


# ================================================================ 11. 합성 (Q1~E1: 블렌딩 없음) · 방법 실행
_FILE_FP = []            # 입력 파일 순서대로의 사진 지문 (덮어쓰기 순서용)
_CAPTURE = None          # 벤치마크용: 합성 없이 변환만 받음


def _fp(im):
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def _file_rank(imgs):
    """정합 과정에서 사진 순서가 바뀌어도 지문으로 원래 파일 순서를 찾음"""
    if not _FILE_FP:
        return list(range(len(imgs)))
    R = np.array(_FILE_FP)
    return [int(np.argmin(np.abs(R - _fp(im)).mean(1))) for im in imgs]


def compose_parts(imgs, fwds, invs):
    """블렌딩 없음: 입력 순서대로 덮어씀"""
    if _CAPTURE is not None:
        _CAPTURE.update(imgs=list(imgs), fwds=list(fwds), invs=list(invs))
        return {"imgs": imgs, "fwds": fwds, "invs": invs, "out": None}
    tiles, (W, H), _ = render(imgs, fwds, invs)
    rank = _file_rank(imgs)
    pano = np.zeros((H, W, 3), np.uint8)
    label = np.full((H, W), -1, np.int32)
    for k in sorted(range(len(tiles)), key=lambda i: rank[i]):
        x0, y0, wimg, v = tiles[k]
        h, w = v.shape
        pano[y0:y0 + h, x0:x0 + w][v] = wimg[v]
        label[y0:y0 + h, x0:x0 + w][v] = k
    ys, xs = np.where(label >= 0)
    return {"imgs": imgs, "fwds": fwds, "invs": invs, "tiles": tiles, "label": label,
            "out": pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()}


class _Setting:
    """방법 번호의 설정(특징점 종류, 투영)을 실행하는 동안만 적용"""
    def __init__(self, cfg, problem):
        self.new = {"FEATURES": cfg.get("features", "sift"), "PROJ": "spherical" if PROJECTION[problem] == "spherical" else "cylindrical"}
        if PROJECTION[problem] == "planar":
            self.new.update(to_cyl=_plane_fwd, from_cyl=_plane_inv)

    def __enter__(self):
        g = globals()
        self.old = {k: g[k] for k in self.new}
        g.update(self.new)

    def __exit__(self, *a):
        globals().update(self.old)


def _stitch_ours(problem, cfg, imgs):
    kind = cfg["kind"]
    if kind == "rew":
        return stitch_q2_parts(imgs)                 # Q1 은 _Setting 이 평면 변환을 넣어 둠
    if kind == "ba":
        pairs = match_pairs_local(imgs, bl_detect, local=True)[:2] if cfg.get("local") else match_all_pairs(imgs)
        dense = cfg.get("dense", False)
        sub, fwds, invs, info = align_ba_rew_mid(imgs, pairs, cfg["rew"], dense=dense, dense_cap=1.0 if dense else None)
        return compose_parts(sub, fwds, invs)
    if kind == "e2":
        return stitch_e2_parts(imgs, cfg["stage"])
    if kind == "homography":
        return _stitch_b0_homography(imgs)
    raise ValueError(kind)


def _stitch_b0_homography(images):
    """Q1 ②: baseline Q1 절차 + 우리 특징점"""
    H01 = _b0_homography_ours(images)
    if H01 is None:
        raise ValueError("Q1 homography failed")
    h21 = np.linalg.inv(H01)
    c2 = cv2.perspectiveTransform(_b0_image_corners(images[1]).reshape(-1, 1, 2).astype(np.float64), h21).reshape(-1, 2)
    allc = np.vstack([_b0_image_corners(images[0]), c2])
    low, high = np.floor(allc.min(0)), np.ceil(allc.max(0))
    W, H = map(int, high - low)
    T = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]], np.float64)
    warped, masks = [], []
    for im, M in zip(images, [T, T @ h21]):
        warped.append(cv2.warpPerspective(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
        masks.append(cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST))
    out, _, _ = _b0_hard_composite(warped, masks, [(0, 0), (0, 0)], label="Q1")
    tiles, label = _b0_tiles(warped, masks, [(0, 0), (0, 0)])
    return {"out": out, "tiles": tiles, "label": label, "imgs": list(images),
            "fwds": [lambda p: np.asarray(p, np.float64), lambda p: cv2.perspectiveTransform(np.asarray(p, np.float64).reshape(-1, 1, 2), h21).reshape(-1, 2)]}


def _b0_homography_ours(images):
    """영상1 → 영상2 호모그래피 (baseline Q1 절차 + 우리 detect/match)"""
    work, sc = _b0_working_images(images)
    p0, p1 = match(detect(work[0]), detect(work[1]))
    if len(p0) < B0_MIN_INLIERS:
        return None
    Hw, m = cv2.findHomography(p0, p1, cv2.RANSAC, B0_Q1_RANSAC_PX, maxIters=5000, confidence=0.995)
    if Hw is None or m is None or int(m.sum()) < B0_MIN_INLIERS:
        return None
    H01 = np.linalg.inv(sc[1]) @ Hw @ sc[0]
    if not np.isfinite(H01).all() or np.linalg.matrix_rank(H01) < 3:
        return None
    c = np.column_stack([_b0_image_corners(images[1]), np.ones(4)]) @ np.linalg.inv(H01).T
    if np.min(np.abs(c[:, 2])) < 1e-8 or not (np.all(c[:, 2] > 0) or np.all(c[:, 2] < 0)):
        return None                                   # baseline 과 같은 지평선 검사
    return H01


def _b0_tiles(warped, masks, corners, select=None):
    """baseline 워핑 영상·마스크 -> (tiles, label)"""
    x0 = min(int(c[0]) for c in corners)
    y0 = min(int(c[1]) for c in corners)
    tiles = [(int(c[0]) - x0, int(c[1]) - y0, w, m > 0) for w, m, c in zip(warped, masks, corners)]
    W = max(t[0] + t[3].shape[1] for t in tiles)
    H = max(t[1] + t[3].shape[0] for t in tiles)
    label = np.full((H, W), -1, np.int32)
    for k, (x, y, _, v) in enumerate(tiles):
        s = v if select is None else select[k] > 0
        label[y:y + s.shape[0], x:x + s.shape[1]][s] = k
    return tiles, label


def _stitch_baseline(problem, inputs):
    """baseline 실행. 평가용으로 덮어쓰기 직전 영상·마스크와 각 사진 → 합성 좌표 변환도 돌려줌"""
    _B0_LAST.clear()
    cv2.setRNGSeed(0)
    out = {"Q1": b0_q1, "Q2": b0_q2, "Q3": b0_q3, "E1": b0_e1, "E2": b0_e2}[problem](list(inputs))
    if problem == "E2":
        corrected, selected, masks, corners = _B0_LAST["e2"]
        tiles, label = _b0_tiles(corrected, masks, corners, selected)
    else:
        tiles, label = _b0_tiles(*_B0_LAST["layers"])
    if problem == "Q1":
        h21 = _B0_LAST["h21"]
        fwds = [lambda p: np.asarray(p, np.float64), lambda p: cv2.perspectiveTransform(np.asarray(p, np.float64).reshape(-1, 1, 2), h21).reshape(-1, 2)]
    else:                                             # 합성 면 좌표 (Q2 는 원통 재투영 전)
        warper, Ks, Rs = _B0_LAST["warp"]
        fwds = [lambda p, K=K, R=R: np.array([warper.warpPoint((float(x), float(y)), K, R) for x, y in p]).reshape(-1, 2) for K, R in zip(Ks, Rs)]
    return {"out": out, "tiles": tiles, "label": label, "fwds": fwds}


def _load(inputs):
    """경로 → 합성용 크기(긴 변 1400px)로 읽은 영상. 배열이면 크기만 맞춤"""
    if all(isinstance(x, np.ndarray) for x in inputs):
        return [_resize_max(im, COMPOSE_MAX_SIDE) for im in inputs]
    return load_images(inputs)[0]


def stitch_parts(problem, inputs, method=None):
    """문제·방법 번호로 정합 → dict(out, tiles, label, imgs, fwds ...). inputs: 경로 또는 영상 배열"""
    method = METHOD[problem] if method is None else method
    cfg = METHODS[problem][method]
    if cfg["kind"] == "baseline":
        return _stitch_baseline(problem, inputs)
    if cfg["kind"] == "homography":                   # baseline 과 같은 입력 (원본 크기)
        imgs = [x if isinstance(x, np.ndarray) else cv2.imread(x) for x in inputs]
    else:
        imgs = _load(inputs)
        _FILE_FP[:] = [_fp(im) for im in imgs]
    cv2.setRNGSeed(0)
    with _Setting(cfg, problem):
        return _stitch_ours(problem, cfg, imgs)


def _concat_fallback(imgs):
    hmin = min(im.shape[0] for im in imgs)
    return np.hstack([cv2.resize(im, (max(1, round(im.shape[1] * hmin / im.shape[0])), hmin)) for im in imgs])


def _safe(fn):
    """정합 실패 시 입력을 이어 붙인 영상을 반환"""
    def wrapper(img_paths):
        try:
            return fn(img_paths)
        except Exception:
            return _concat_fallback(_load(img_paths))
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def _prepare(img_paths):
    imgs = _load(img_paths)
    _FILE_FP[:] = [_fp(im) for im in imgs]
    cv2.setRNGSeed(0)
    return imgs


# ================================================================ 제출 형식 함수
@_safe
def q1(img_paths):
    """Q1: 평면 2장. SIFT + 엣지 교점 -> homography -> REW -> 덮어쓰기."""
    if METHOD["Q1"] != 5:
        return stitch_parts("Q1", img_paths)["out"]           # 비교용 방법 (1 = baseline)
    imgs = _prepare(img_paths)
    with _Setting(METHODS["Q1"][5], "Q1"):                    # 평면 좌표, 엣지 교점 대응점 사용
        pairs = match_all_pairs(imgs)                         # SIFT -> ratio 0.75 -> RANSAC -> 판정식
        sub, fwds, invs, _ = align_rew(imgs, pairs)           # 전역 homography + REW(TPS) 보정
        return compose_parts(sub, fwds, invs)["out"]          # 블렌딩 없이 덮어쓰기


@_safe
def q2(img_paths):
    """Q2: 원통 3장. 원통 투영 -> SIFT -> homography -> REW -> 덮어쓰기."""
    if METHOD["Q2"] != 4:
        return stitch_parts("Q2", img_paths)["out"]
    imgs = _prepare(img_paths)
    with _Setting(METHODS["Q2"][4], "Q2"):
        pairs = match_all_pairs(imgs)                         # SIFT -> ratio 0.75 -> RANSAC -> 판정식
        sub, fwds, invs, _ = align_rew(imgs, pairs)           # 원통 좌표에서 homography + REW, 트리로 연결
        return compose_parts(sub, fwds, invs)["out"]


@_safe
def q3(img_paths):
    """Q3: 2x4 격자 8장, 원통. 국소 일관성 매칭 -> BA -> DIS + 중간점 TPS -> 덮어쓰기."""
    if METHOD["Q3"] != 7:
        return stitch_parts("Q3", img_paths)["out"]
    imgs = _prepare(img_paths)
    with _Setting(METHODS["Q3"][7], "Q3"):
        pairs = match_pairs_local(imgs, bl_detect, local=True)[:2]   # 모든 쌍 매칭 + 국소 일관성 검사
        sub, fwds, invs, _ = align_ba_rew_mid(imgs, pairs, dense=True, dense_cap=1.0)   # BA -> 원통 -> TPS
        return compose_parts(sub, fwds, invs)["out"]


@_safe
def e1(img_paths):
    """E1: Q3 의 8장, 구면. Q3 와 같은 정합."""
    if METHOD["E1"] != 7:
        return stitch_parts("E1", img_paths)["out"]
    imgs = _prepare(img_paths)
    with _Setting(METHODS["E1"][7], "E1"):                    # 구면 투영
        pairs = match_pairs_local(imgs, bl_detect, local=True)[:2]
        sub, fwds, invs, _ = align_ba_rew_mid(imgs, pairs, dense=True, dense_cap=1.0)
        return compose_parts(sub, fwds, invs)["out"]


@_safe
def e2(img_paths):
    """E2: 8장, 구면. E1 정합 + gain·비네팅 보정 + 어긋남 회피 seam + multi-band."""
    if METHOD["E2"] != 5:
        return stitch_parts("E2", img_paths)["out"]
    imgs = _prepare(img_paths)
    with _Setting(METHODS["E2"][5], "E2"):
        pairs = match_pairs_local(imgs, bl_detect, local=True)[:2]
        sub, fwds, invs, _ = align_ba_rew_mid(imgs, pairs, dense=True, dense_cap=1.0)
        tiles, size, _ = render(sub, fwds, invs)
        wtiles, _, _ = render([weight_image(im) for im in sub], fwds, invs)
        rtiles, _, _ = render([radius_image(im) for im in sub], fwds, invs)
        gained, _ = exposure_vignetting(tiles, rtiles)        # 채널별 gain + 비네팅
        label = seam_labels_s(gained, size, wtiles)            # 색·어긋남·물체 경계를 피하는 seam
        pano, mask = multiband_label(gained, size, label)      # 5단계 multi-band
    ys, xs = np.where(mask)
    return pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ---------------------------------------------------------------------------
# Direct CLI: 결과 저장 · 평가 · 벤치마크. q1~e2 는 이 아래를 쓰지 않음.
# ---------------------------------------------------------------------------
import argparse
import concurrent.futures
import glob
import json
import os
import time

_DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")   # 평가 표 (results/ 에는 PNG 만)
PROBLEM_DATA = {"Q1": ("Q1", 2), "Q2": ("Q2", 3), "Q3": ("Q3", 8), "E1": ("Q3", 8), "E2": ("Q3", 8)}
# 정합 오차를 재는 겹치는 쌍 (1부터)
EVAL_PAIRS = {"Q1": [(1, 2)], "Q2": [(1, 2), (2, 3)],
              "Q3": [(1, 2), (1, 5), (1, 6), (2, 3), (2, 6), (2, 7), (3, 4), (3, 6), (3, 7), (4, 7), (4, 8), (5, 6), (6, 7), (7, 8)]}
EVAL_PAIRS["E1"] = EVAL_PAIRS["E2"] = EVAL_PAIRS["Q3"]


def parse_cases(args_case):
    """'all' / 'Q3' (그 문제의 모든 방법) / 'Q3-7' → [(문제, 번호)]"""
    out = []
    for c in args_case:
        c = c.upper()
        if c == "ALL":
            out += [(q, m) for q in METHODS for m in METHODS[q]]
        elif c in METHODS:
            out += [(c, m) for m in METHODS[c]]
        elif c == "FINAL":
            out += list(METHOD.items())
        else:
            q, m = c.split("-")
            if int(m) not in METHODS[q]:
                raise SystemExit(f"{q} 에 방법 {m} 이 없습니다 (python stitching.py --task list)")
            out.append((q, int(m)))
    return out


def _input_paths(q, folder=None):
    d, n = PROBLEM_DATA[q]
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", d)
    ps = sorted((p for p in glob.glob(os.path.join(folder, "*")) if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")),
                key=lambda p: [int(t) if t.isdigit() else t.lower() for t in __import__("re").split(r"(\d+)", os.path.basename(p))])
    if len(ps) != n:
        print(f"[경고] {q}: 이미지 {n}장을 기대했지만 {len(ps)}장 ({folder})")
    return ps


# ---------------------------------------------------------------- 평가 (프로젝트 사진)
_DIS = None


def boundary_metrics(tiles, label):
    """경계 띠의 밝기 차이와 어긋남(DIS flow)"""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    col, mis = [], []
    for a in range(len(tiles)):
        for b in range(a + 1, len(tiles)):
            xa, ya, wa, va = tiles[a]
            xb, yb, wb, vb = tiles[b]
            X0, Y0 = max(xa, xb), max(ya, yb)
            X1, Y1 = min(xa + va.shape[1], xb + vb.shape[1]), min(ya + va.shape[0], yb + vb.shape[0])
            if X1 - X0 < 10 or Y1 - Y0 < 10:
                continue
            L = label[Y0:Y1, X0:X1]
            e = np.zeros(L.shape, bool)
            e[:, :-1] |= ((L[:, :-1] == a) & (L[:, 1:] == b)) | ((L[:, :-1] == b) & (L[:, 1:] == a))
            e[:-1, :] |= ((L[:-1, :] == a) & (L[1:, :] == b)) | ((L[:-1, :] == b) & (L[1:, :] == a))
            A = wa[Y0 - ya:Y1 - ya, X0 - xa:X1 - xa]
            Bc = wb[Y0 - yb:Y1 - yb, X0 - xb:X1 - xb]
            Va, Vb = va[Y0 - ya:Y1 - ya, X0 - xa:X1 - xa], vb[Y0 - yb:Y1 - yb, X0 - xb:X1 - xb]
            O = Va & Vb
            if e.sum() < 20:
                continue
            # 경계에서 20px 띠, 겹침 테두리에서 8px 안쪽만
            band = (cv2.dilate(e.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & (cv2.erode(O.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
            if band.sum() < 50:
                continue
            Af, Bf = A.copy(), Bc.copy()
            Af[~Va] = Bc[~Va]
            Bf[~Vb] = A[~Vb]
            ga, gb = cv2.cvtColor(Af, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Bf, cv2.COLOR_BGR2GRAY)
            grad = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(ga.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(ga.astype(np.float32), cv2.CV_32F, 0, 1)), (0, 0), 2)
            Of = O.astype(np.float32)
            wbl = cv2.GaussianBlur(Of, (0, 0), 4)
            Abl = cv2.GaussianBlur(A.astype(np.float32) * Of[..., None], (0, 0), 4) / np.maximum(wbl, 1e-3)[..., None]
            Bbl = cv2.GaussianBlur(Bc.astype(np.float32) * Of[..., None], (0, 0), 4) / np.maximum(wbl, 1e-3)[..., None]
            gbl = cv2.cvtColor(Abl, cv2.COLOR_BGR2GRAY)
            gbl = cv2.magnitude(cv2.Sobel(gbl, cv2.CV_32F, 1, 0), cv2.Sobel(gbl, cv2.CV_32F, 0, 1))
            flat = band & (gbl < 20)
            if flat.sum():
                col.append(np.abs(Abl - Bbl).mean(2)[flat])
            tex = band & (grad > 15)
            if tex.sum() > 10:
                f, bw = _DIS.calc(ga, gb, None), _DIS.calc(gb, ga, None)
                gy, gx = np.mgrid[0:ga.shape[0], 0:ga.shape[1]].astype(np.float32)
                bx = cv2.remap(bw[..., 0], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
                by = cv2.remap(bw[..., 1], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
                ok = tex & (np.hypot(f[..., 0] + bx, f[..., 1] + by) < 1.0)
                mis.append(np.hypot(f[..., 0], f[..., 1])[ok])
    col = np.concatenate(col) if col else np.array([np.nan])
    mis = np.concatenate(mis) if mis else np.array([np.nan])
    return dict(col_med=float(np.median(col)), mis_med=float(np.median(mis)), mis_p95=float(np.percentile(mis, 95)), mis_gt2=float((mis > 2).mean()))


def _orb_pairs(imgs, pairs):
    """정합에 쓰지 않은 독립 대응점: ORB 30000 → ratio 0.75 → RANSAC 10px, 인라이어 30 이상인 쌍만"""
    orb = cv2.ORB_create(30000, fastThreshold=5)
    kd = [orb.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in imgs]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ev = {}
    for i, j in pairs:
        (ka, da), (kb, db) = kd[i], kd[j]
        ms = [m for m, m2 in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * m2.distance]
        pa = np.float64([ka[m.queryIdx].pt for m in ms])
        pb = np.float64([kb[m.trainIdx].pt for m in ms])
        H, msk = cv2.findHomography(pb, pa, cv2.RANSAC, 10.0)
        if H is not None and msk.sum() >= 30:
            msk = msk.ravel().astype(bool)
            ev[(i, j)] = (pa[msk], pb[msk])
    return ev


def alignment_error(fwds, ev):
    """독립 대응점을 각 사진의 변환으로 합성 좌표에 올렸을 때 거리 (px)"""
    E = np.concatenate([np.linalg.norm(fwds[i](pa) - fwds[j](pb), axis=1) for (i, j), (pa, pb) in ev.items()])
    return dict(err_med=float(np.median(E)), err_p90=float(np.percentile(E, 90)))


def evaluate_case(q, m, paths):
    """프로젝트 사진 평가 (입력 긴 변 1400px)"""
    imgs = _load(paths)
    cfg = METHODS[q][m]
    t = time.time()
    with _Setting(cfg, q):                            # Q1 평면 변환·투영 설정이 정합 오차 계산까지 유지되게
        p = stitch_parts(q, imgs if cfg["kind"] != "homography" else [cv2.imread(x) for x in paths], m)
        sec = time.time() - t
        r = dict(case=f"{q}-{m}", name=cfg["name"], seconds=round(sec, 1), size=f"{p['out'].shape[1]}x{p['out'].shape[0]}")
        r.update(boundary_metrics(p["tiles"], p["label"]))
        if cfg["kind"] in ("baseline", "homography"):
            fw = p["fwds"]
        else:
            rank = _file_rank(p["imgs"])
            byfile = {rank[k]: f for k, f in enumerate(p["fwds"])}
            fw = [byfile.get(i) for i in range(len(imgs))]
        ev = _orb_pairs(imgs, [(i - 1, j - 1) for i, j in EVAL_PAIRS[q] if fw[i - 1] is not None and fw[j - 1] is not None])
        if ev:
            r.update(alignment_error(fw, ev))
    return r, p["out"]


# ---------------------------------------------------------------- 벤치마크 (HPatches · UDIS-D, 두 영상 경로)
def pair_transform(q, m, paths):
    """두 영상에 방법을 적용 → (T: 영상1 원본 좌표 → 영상2 원본 좌표, Tinv) 또는 None."""
    global _CAPTURE
    cfg = METHODS[q][m]
    if cfg["kind"] in ("baseline", "homography"):
        images = [cv2.imread(p) for p in paths]
        if cfg["kind"] == "homography":
            with _Setting(cfg, q):
                H01 = _b0_homography_ours(images)
        else:
            H01 = _b0_pair_homography(images)
        if H01 is None:
            return None
        Hi = np.linalg.inv(H01)
        ph = lambda p, M: cv2.perspectiveTransform(np.asarray(p, np.float64).reshape(-1, 1, 2), M).reshape(-1, 2)
        return (lambda p: ph(p, H01)), (lambda p: ph(p, Hi))
    if cfg["kind"] == "e2":                           # 블렌딩은 변환에 영향 없음 -> E2 정합 = E1 ⑦ (구면)
        q, cfg = "E1", METHODS["E1"][7]
    imgs = _load(paths)
    sc = [im.shape[0] / cv2.imread(p).shape[0] for im, p in zip(imgs, paths)]
    _FILE_FP[:] = [_fp(im) for im in imgs]
    cv2.setRNGSeed(0)
    setting = _Setting(cfg, q)
    _CAPTURE = {}
    try:
        with setting:
            _stitch_ours(q, cfg, imgs)
    except Exception:
        return None
    finally:
        cap, _CAPTURE = _CAPTURE, None
    if len(cap.get("imgs", [])) < 2:
        return None
    pos = {}
    for t, im in enumerate(cap["imgs"]):
        for k, ref in enumerate(imgs):
            if im is ref or (im.shape == ref.shape and np.array_equal(im, ref)):
                pos[k] = t
    if 0 not in pos or 1 not in pos:
        return None
    F, I = cap["fwds"], cap["invs"]

    def make(a, b):
        def T(p):
            with setting:                             # 변환 함수는 호출할 때 투영 설정을 읽음
                P = F[pos[a]](np.asarray(p, np.float64) * sc[a])
                x, y = I[pos[b]](P[:, 0].copy(), P[:, 1].copy())
            out = np.column_stack([x, y]) / sc[b]
            out[(np.abs(out) > 1e5).any(1)] = np.nan
            return out
        return T
    return make(0, 1), make(1, 0)


def _b0_pair_homography(images):
    """baseline Q1 호모그래피 (영상1 -> 영상2)"""
    _B0_LAST.clear()
    cv2.setRNGSeed(0)
    try:
        b0_q1(list(images))
    except Exception:
        return None
    h21 = _B0_LAST["h21"]
    warped, masks, _ = _B0_LAST["layers"]
    c0 = _b0_image_corners(images[0])
    c1 = cv2.perspectiveTransform(_b0_image_corners(images[1]).reshape(-1, 1, 2).astype(np.float64), h21).reshape(-1, 2)
    x, y = (-np.floor(np.vstack([c0, c1]).min(0))).astype(int)
    h, w = images[0].shape[:2]
    if warped[1][y:y + h, x:x + w].shape != images[0].shape or np.count_nonzero(masks[1][y:y + h, x:x + w]) < 64:
        return None
    return np.linalg.inv(h21)


def warp_into(src, dst_shape, Tinv):
    h, w = dst_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    p = Tinv(np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64))
    mx = np.nan_to_num(p[:, 0], nan=-1e6).reshape(h, w).astype(np.float32)
    my = np.nan_to_num(p[:, 1], nan=-1e6).reshape(h, w).astype(np.float32)
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out, (mx >= 0) & (mx <= src.shape[1] - 1) & (my >= 0) & (my <= src.shape[0] - 1)


def psnr(a, b, m):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))[m] ** 2).mean())
    return 100.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim(a, b, m, min_centers=0):
    """BGR 채널별, 11×11 Gaussian σ=1.5, 창 전체가 마스크 안인 중심만"""
    inner = cv2.erode(m.astype(np.uint8), np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if inner.sum() < max(min_centers, 1):
        return None
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    vals = []
    for c in range(3):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        mx, my = g(x), g(y)
        sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
        vals.append((((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2)))[inner].mean())
    return float(np.mean(vals))


def bench_hp_pair(job):
    """HPatches 한 쌍 (1번 ↔ k번): 독립점 오차(20×20 격자), Corner error, 정답 겹침 영역 PSNR·SSIM"""
    d, k, cases = job
    cv2.setNumThreads(1)
    pa, pb = os.path.join(d, "1.ppm"), os.path.join(d, f"{k}.ppm")
    A, Bm = cv2.imread(pa), cv2.imread(pb)
    H = np.loadtxt(os.path.join(d, f"H_1_{k}"))
    h, w = A.shape[:2]
    gx, gy = np.meshgrid(np.linspace(10, w - 11, 20), np.linspace(10, h - 11, 20))
    P = np.column_stack([gx.ravel(), gy.ravel()])
    Qgt = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2)
    inside = (Qgt[:, 0] >= 0) & (Qgt[:, 0] <= Bm.shape[1] - 1) & (Qgt[:, 1] >= 0) & (Qgt[:, 1] <= Bm.shape[0] - 1)
    corners = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    Cgt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    hb, wb = Bm.shape[:2]
    yy, xx = np.mgrid[0:hb, 0:wb]
    back = cv2.perspectiveTransform(np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64).reshape(-1, 1, 2), np.linalg.inv(H)).reshape(hb, wb, 2)
    gt_ov = (back[..., 0] >= 0) & (back[..., 0] <= w - 1) & (back[..., 1] >= 0) & (back[..., 1] <= h - 1)
    seq = os.path.basename(d)
    rows = []
    for q, m in cases:
        r = dict(set="hp", seq=seq, k=k, group="illum" if seq.startswith("i_") else "view", case=f"{q}-{m}")
        t = time.time()
        res = pair_transform(q, m, [pa, pb])
        r["seconds"] = time.time() - t
        r["ok"] = res is not None
        if res is None:
            Tinv = lambda p: np.asarray(p, np.float64).copy()       # 광도 평가는 항등변환으로 대체
        else:
            T, Tinv = res
            e = np.linalg.norm(T(P[inside]) - Qgt[inside], axis=1)
            e = np.where(np.isfinite(e), e, 1e4)
            r.update(pt_med=float(np.median(e)), pt_rmse=float(np.sqrt((e ** 2).mean())), pt_p95=float(np.percentile(e, 95)))
            ce = np.linalg.norm(T(corners) - Cgt, axis=1)
            r["corner"] = float(np.mean(np.where(np.isfinite(ce), ce, 1e4)))
        warped, _ = warp_into(A, Bm.shape, Tinv)
        r["psnr"], r["ssim"] = psnr(warped, Bm, gt_ov), ssim(warped, Bm, gt_ov)
        rows.append(r)
    return rows


def bench_udis_pair(job):
    """UDIS-D 한 쌍: 공통 겹침 영역 PSNR·SSIM"""
    root, name, cases = job
    cv2.setNumThreads(1)
    p1, p2 = os.path.join(root, "input1", name), os.path.join(root, "input2", name)
    I1, I2 = cv2.imread(p1), cv2.imread(p2)
    W, rows = {}, []
    for q, m in cases:
        r = dict(set="udis", name=name, case=f"{q}-{m}")
        t = time.time()
        res = pair_transform(q, m, [p1, p2])
        r["seconds"] = time.time() - t
        r["ok"] = res is not None
        warped, valid = warp_into(I2, I1.shape, res[0] if res is not None else (lambda p: np.asarray(p, np.float64).copy()))
        W[r["case"]] = warped, valid
        r["psnr_ind"] = psnr(warped, I1, valid) if valid.sum() else None
        r["ssim_ind"] = ssim(warped, I1, valid, 256)
        rows.append(r)
    common = np.logical_and.reduce([v for _, v in W.values()])
    for r in rows:
        s = ssim(W[r["case"]][0], I1, common, 256)
        r.update(area=float(common.mean()), ssim=s, psnr=psnr(W[r["case"]][0], I1, common) if s is not None else None)
    return rows


def summarize(rows, cases):
    keys = [f"{q}-{m}" for q, m in cases]
    out = {}
    hp = [r for r in rows if r["set"] == "hp"]
    for name, sub in (("hp", hp), ("hp_illum", [r for r in hp if r["group"] == "illum"]), ("hp_view", [r for r in hp if r["group"] == "view"])):
        if not sub:
            continue
        out[name] = {}
        for k in keys:
            R = [r for r in sub if r["case"] == k]
            ok = [r for r in R if r["ok"]]
            ce = np.array([r["corner"] if r["ok"] else np.inf for r in R])
            x = dict(n=len(R), ok=len(ok), psnr=float(np.mean([r["psnr"] for r in R])),
                     ssim=float(np.mean([r["ssim"] for r in R if r["ssim"] is not None])), sec=float(np.median([r["seconds"] for r in R])))
            if ok:
                x.update(pt_med=float(np.median([r["pt_med"] for r in ok])), pt_rmse=float(np.mean([r["pt_rmse"] for r in ok])),
                         pt_p95=float(np.median([r["pt_p95"] for r in ok])), corner_med=float(np.median([r["corner"] for r in ok])))
            for t in (1, 3, 5, 10):
                x[f"succ{t}"] = float(100 * np.mean(ce <= t))
                x[f"auc{t}"] = float(100 * np.mean(np.maximum(1 - ce / t, 0)))
            out[name][k] = x
    ud = [r for r in rows if r["set"] == "udis"]
    if ud:
        out["udis"] = {}
        for k in keys:
            R = [r for r in ud if r["case"] == k]
            sc = [r for r in R if r["psnr"] is not None]
            ind = [r for r in R if r["psnr_ind"] is not None and r["ssim_ind"] is not None]
            out["udis"][k] = dict(n=len(R), ok=sum(r["ok"] for r in R), psnr=float(np.mean([r["psnr"] for r in sc])),
                                  ssim=float(np.mean([r["ssim"] for r in sc])), area=float(100 * np.mean([r["area"] for r in R])),
                                  psnr_ind=float(np.mean([r["psnr_ind"] for r in ind])), ssim_ind=float(np.mean([r["ssim_ind"] for r in ind])),
                                  sec=float(np.median([r["seconds"] for r in R])))
    return out


def _md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def benchmark_report(S, cases):
    name = {f"{q}-{m}": f"{q}-{m} {METHODS[q][m]['name']}" + (" (정합 = E1 ⑦)" if q == "E2" else "") for q, m in cases}
    parts = ["# HPatches · UDIS-D 벤치마크", "", "두 영상에 그대로 적용 (baseline 은 Q1 경로). 오차는 성공 쌍, 성공률·AUC 는 전체 쌍 기준.", ""]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    if "hp" in S:
        parts += ["## HPatches 기하 정확도", "", _md_table(
            ["방법", "성공", "독립점 Median px", "RMSE px", "P95 px", "Corner Median px", "성공@3px %", "성공@5px %", "AUC@5px %"],
            [[name[k], f"{v['ok']}/{v['n']}", f(v.get("pt_med")), f(v.get("pt_rmse")), f(v.get("pt_p95")), f(v.get("corner_med")),
              f(v["succ3"], 2), f(v["succ5"], 2), f(v["auc5"], 2)] for k, v in S["hp"].items()]), ""]
        if "hp_illum" in S and "hp_view" in S:
            parts += ["### 조명·시점 그룹별", "", _md_table(["방법", "조명: 성공@5 %", "조명: Median px", "시점: 성공@5 %", "시점: Median px"],
                      [[name[k], f(S["hp_illum"][k]["succ5"], 2), f(S["hp_illum"][k].get("pt_med")), f(S["hp_view"][k]["succ5"], 2),
                        f(S["hp_view"][k].get("pt_med"))] for k in S["hp"]]), ""]
        parts += ["### 광도 (정답 겹침 영역)", "", _md_table(["방법", "PSNR dB", "SSIM"], [[name[k], f(v["psnr"]), f(v["ssim"], 4)] for k, v in S["hp"].items()]), ""]
    if "udis" in S:
        parts += ["## UDIS-D", "", "비교한 방법들의 공통 겹침 영역 (실패는 항등변환)", "", _md_table(
            ["방법", "성공", "PSNR dB", "SSIM", "평가 면적 %", "개별 겹침 PSNR", "개별 겹침 SSIM"],
            [[name[k], f"{v['ok']}/{v['n']}", f(v["psnr"]), f(v["ssim"], 4), f(v["area"], 2), f(v["psnr_ind"]), f(v["ssim_ind"], 4)] for k, v in S["udis"].items()]), ""]
    return "\n".join(parts)


def run_benchmark(args, cases):
    root = args.data
    jobs = []
    if args.dataset in ("hpatches", "all"):
        seqs = sorted(glob.glob(os.path.join(root, "hpatches-sequences-release", "*")))
        if not seqs:
            raise SystemExit(f"{root}/hpatches-sequences-release 가 없습니다. 받는 곳: "
                             "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz")
        jobs += [(bench_hp_pair, (d, k, cases)) for d in seqs for k in range(2, 7)][:args.limit or None]
    if args.dataset in ("udis", "all"):
        udis = os.path.join(root, "testing")
        names = sorted(set(map(os.path.basename, glob.glob(os.path.join(udis, "input1", "*.jpg")))) &
                       set(map(os.path.basename, glob.glob(os.path.join(udis, "input2", "*.jpg")))))
        if not names:
            raise SystemExit(f"{udis}/input1, input2 가 없습니다. UDIS-D 받는 곳: https://github.com/nie-lang/UnsupervisedDeepImageStitching")
        jobs += [(bench_udis_pair, (udis, n, cases)) for n in names][:args.limit or None]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fn, job) for fn, job in jobs]
        for fu in concurrent.futures.as_completed(futs):
            rows += fu.result()
    S = summarize(rows, cases)
    md = benchmark_report(S, cases)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fo:
        fo.write(md + "\n")
    if args.rows:
        with open(args.rows, "w", encoding="utf-8") as fo:
            fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print("[OK]")


def main():
    ap = argparse.ArgumentParser(description="이미지 정합: 결과 저장 · 평가 · 벤치마크")
    ap.add_argument("--task", choices=["list", "stitch", "evaluate", "benchmark"], default="stitch")
    ap.add_argument("--case", nargs="+", default=["final"], help="final(기본, METHOD 값) / all / Q3 / Q3-7 ...")
    ap.add_argument("--input", help="입력 폴더 (문제 하나일 때, 기본: data/Q1 · data/Q2 · data/Q3)")
    ap.add_argument("--out", default="results", help="결과 사진 폴더")
    ap.add_argument("--report", help="평가·벤치마크 표 저장 경로 (기본: docs/hj_mission_results.md, docs/hj_benchmark_results.md)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(_DOCS), "data", "benchmarks"), help="벤치마크 데이터 (hpatches-sequences-release/, testing/)")
    ap.add_argument("--dataset", choices=["hpatches", "udis", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="데이터셋마다 앞에서 N쌍만 (시험 실행)")
    ap.add_argument("--rows", help="벤치마크 쌍별 결과 (JSON lines) 저장 경로")
    args = ap.parse_args()
    if args.task == "list":
        for q, ms in METHODS.items():
            print(q, f"({PROJECTION[q]})")
            for m, c in ms.items():
                print(f"  {q}-{m}  {c['name']}" + ("   ← 기본값 (METHOD)" if METHOD[q] == m else ""))
        return
    cases = parse_cases(args.case)
    if args.task == "benchmark":
        args.report = args.report or os.path.join(_DOCS, "hj_benchmark_results.md")
        return run_benchmark(args, cases)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for q, m in cases:
        paths = _input_paths(q, args.input)
        if args.task == "stitch":
            out = stitch_parts(q, paths, m)["out"]
            cv2.imwrite(os.path.join(args.out, f"{q}_m{m}.png"), out)
            print("[OK]", flush=True)
        else:
            r, out = evaluate_case(q, m, paths)
            cv2.imwrite(os.path.join(args.out, f"{q}_m{m}.png"), out)
            rows.append(r)
            print("[OK]", flush=True)
    if args.task == "evaluate":
        f = lambda v, d=2: "-" if v is None or v != v else f"{v:.{d}f}"
        md = "\n".join(["# 프로젝트 사진 평가", "",
                        "어긋남: 경계 20px 띠의 DIS flow · 정합 오차: 독립 ORB 대응점 거리 · 입력 긴 변 1400px", "",
                        _md_table(["방법", "어긋남 >2px", "어긋남 중앙 px", "어긋남 95% px", "밝기 차이 중앙", "정합 오차 중앙 px", "정합 오차 90% px", "크기", "시간 s"],
                                  [[f"{r['case']} {r['name']}", f"{100 * r['mis_gt2']:.1f}%", f(r["mis_med"]), f(r["mis_p95"]), f(r["col_med"], 1),
                                    f(r.get("err_med")), f(r.get("err_p90")), r["size"], r["seconds"]] for r in rows])])
        path = args.report or os.path.join(_DOCS, "hj_mission_results.md")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(md + "\n")


if __name__ == "__main__":
    main()
