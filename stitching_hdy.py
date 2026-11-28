"""Small feature-based panorama candidate for the five assignment missions.

Shared path: SIFT -> ratio/mutual matching -> RANSAC -> joint camera rotations
-> cylindrical/spherical warp -> composition. Q1 uses a planar homography.

The joint-rotation formulation follows the teammate's `stitching_Lsb.py`
experiment (jungejblue/IS_ws, inspected 2026-11-28). This is a new,
simplified implementation for comparison, not a full reproduction of the
Brown--Lowe system. It uses no LPAM seam code or external Python imports.
"""

import cv2
import numpy as np


def _read(paths, expected):
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} images; received {len(paths)}")
    images = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        scale = min(1.0, 1400.0 / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        images.append(image)
    return images


def _features(images):
    sift = cv2.SIFT_create(nfeatures=6000)
    result = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 12:
            raise ValueError("Too few SIFT features")
        result.append((keypoints, descriptors))
    return result


def _matched_points(first, second, matcher):
    ka, da = first
    kb, db = second
    forward = matcher.knnMatch(da, db, k=2)
    backward = matcher.knnMatch(db, da, k=2)
    reverse = {m.queryIdx: m.trainIdx for row in backward if len(row) == 2
               for m, n in [row] if m.distance < 0.78 * n.distance}
    matches = [m for row in forward if len(row) == 2
               for m, n in [row] if m.distance < 0.78 * n.distance
               and reverse.get(m.trainIdx) == m.queryIdx]
    if len(matches) < 12:
        return None
    source = np.float32([ka[m.queryIdx].pt for m in matches])
    target = np.float32([kb[m.trainIdx].pt for m in matches])
    return source, target


def _edges(images, pairs):
    features = _features(images)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),
                                    dict(checks=64))
    edges = []
    for i, j in pairs:
        points = _matched_points(features[i], features[j], matcher)
        if points is None:
            continue
        source, target = points
        cv2.setRNGSeed(42)
        homography, keep = cv2.findHomography(
            source, target, cv2.RANSAC, 4.0,
            maxIters=5000, confidence=0.995)
        if homography is None or keep is None or int(keep.sum()) < 12:
            continue
        selected = keep.ravel() != 0
        source, target = source[selected], target[selected]
        if not np.isfinite(homography).all():
            continue
        edges.append((i, j, source, target, homography))
    return edges


def _compose(layers, blend):
    corners = np.array([layer[2] for layer in layers], np.int32)
    ends = np.array([layer[2] + layer[1].shape[::-1] for layer in layers])
    origin = corners.min(axis=0)
    width, height = (ends.max(axis=0) - origin).tolist()
    if min(width, height) < 2 or width * height > 22_000_000:
        raise ValueError("Invalid panorama canvas")

    labels = np.full((height, width), -1, np.int16)
    best = np.full((height, width), -1.0, np.float32)
    for index, (image, mask, corner) in enumerate(layers):
        x, y = (corner - origin).tolist()
        valid = mask > 0
        ih, iw = valid.shape
        padded = cv2.copyMakeBorder(valid.astype(np.uint8), 1, 1, 1, 1,
                                     cv2.BORDER_CONSTANT)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        region = np.s_[y:y + ih, x:x + iw]
        take = valid & (distance > best[region])
        best[region][take] = distance[take]
        labels[region][take] = index
    yy, xx = np.nonzero(labels >= 0)
    if len(xx) == 0:
        raise ValueError("No valid panorama pixels")
    crop = np.s_[yy.min():yy.max() + 1, xx.min():xx.max() + 1]

    if not blend:
        output = np.zeros((height, width, 3), np.uint8)
        for index, (image, _, corner) in enumerate(layers):
            x, y = (corner - origin).tolist()
            ih, iw = image.shape[:2]
            region = np.s_[y:y + ih, x:x + iw]
            take = labels[region] == index
            output[region][take] = image[take]
        return output[crop]

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(5)
    blender.prepare((0, 0, width, height))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    for index, (image, mask, corner) in enumerate(layers):
        x, y = (corner - origin).tolist()
        ih, iw = image.shape[:2]
        owned = (labels[y:y + ih, x:x + iw] == index).astype(np.uint8)
        blend_mask = cv2.dilate(owned, kernel) * (mask > 0).astype(np.uint8)
        blender.feed(image.astype(np.int16), blend_mask * 255, (x, y))
    output, valid = blender.blend(None, None)
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[valid == 0] = 0
    return output[crop]


def q1(img_paths):
    """Two planar images: SIFT, RANSAC homography, hard composition."""
    images = _read(img_paths, 2)
    edges = _edges(images, [(0, 1)])
    if len(edges) != 1:
        raise ValueError("Q1 images could not be matched")
    homography = edges[0][4]  # image 1 -> image 2
    first, second = images
    h1, w1 = first.shape[:2]
    h2, w2 = second.shape[:2]
    second_corners = np.float32([[0, 0], [w2 - 1, 0],
                                 [w2 - 1, h2 - 1], [0, h2 - 1]])
    moved = cv2.perspectiveTransform(second_corners[None],
                                     np.linalg.inv(homography))[0]
    bounds = np.vstack(([[0, 0], [w1 - 1, h1 - 1]], moved))
    low = np.floor(bounds.min(axis=0)).astype(int)
    high = np.ceil(bounds.max(axis=0)).astype(int)
    width, height = (high - low + 1).tolist()
    if min(width, height) < 2 or width * height > 22_000_000:
        raise ValueError("Invalid Q1 homography bounds")
    shift = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1]],
                     np.float64)
    layers = []
    for image, transform in ((first, shift),
                             (second, shift @ np.linalg.inv(homography))):
        mask = np.full(image.shape[:2], 255, np.uint8)
        warped = cv2.warpPerspective(image, transform, (width, height))
        valid = cv2.warpPerspective(mask, transform, (width, height),
                                    flags=cv2.INTER_NEAREST)
        layers.append((warped, valid, np.array([0, 0], np.int32)))
    return _compose(layers, blend=False)


def _rotation_from_homography(homography, intrinsic):
    ray_map = np.linalg.inv(intrinsic) @ homography @ intrinsic
    if np.linalg.det(ray_map) < 0:
        ray_map = -ray_map
    u, _, vt = np.linalg.svd(ray_map)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _camera_rotations(edges, images):
    """Fit all camera directions together from matched image pairs."""
    count = len(images)
    shape = images[0].shape[:2]
    if any(im.shape[:2] != shape for im in images):
        raise ValueError("The current camera model requires equal image sizes")
    height, width = shape
    focal_initial = width * 1.1
    intrinsic = np.array([[focal_initial, 0, (width - 1) / 2],
                          [0, focal_initial, (height - 1) / 2],
                          [0, 0, 1]], np.float64)
    relatives = [(i, j, _rotation_from_homography(h, intrinsic), len(a))
                 for i, j, a, b, h in edges]
    anchor = 1 if count > 1 else 0
    initial = {anchor: np.eye(3)}
    while len(initial) < count:
        candidates = [e for e in relatives
                      if (e[0] in initial) != (e[1] in initial)]
        if not candidates:
            raise ValueError("The image matching graph is disconnected")
        i, j, relative, _ = max(candidates, key=lambda e: e[3])
        if i in initial:
            initial[j] = initial[i] @ relative.T
        else:
            initial[i] = initial[j] @ relative

    variable_images = [i for i in range(count) if i != anchor]
    x = np.concatenate(([np.log(focal_initial)],
                        *[cv2.Rodrigues(initial[i])[0].ravel()
                          for i in variable_images]))
    sampled = []
    for i, j, a, b, _ in edges:
        take = np.linspace(0, len(a) - 1, min(len(a), 150), dtype=int)
        sampled.append((i, j, a[take].astype(np.float64),
                        b[take].astype(np.float64)))

    def residual(parameters):
        focal = float(np.exp(parameters[0]))
        rotations = [np.eye(3) for _ in images]
        for position, image_index in enumerate(variable_images):
            rotations[image_index] = cv2.Rodrigues(
                parameters[1 + position * 3:4 + position * 3])[0]
        all_errors = []
        for i, j, a, b in sampled:
            ra = np.column_stack(((a[:, 0] - (width - 1) / 2) / focal,
                                  (a[:, 1] - (height - 1) / 2) / focal,
                                  np.ones(len(a))))
            rb = np.column_stack(((b[:, 0] - (width - 1) / 2) / focal,
                                  (b[:, 1] - (height - 1) / 2) / focal,
                                  np.ones(len(b))))
            ra /= np.linalg.norm(ra, axis=1)[:, None]
            rb /= np.linalg.norm(rb, axis=1)[:, None]
            error = (ra @ rotations[i].T - rb @ rotations[j].T)
            all_errors.append((error * focal / np.sqrt(len(a))).ravel())
        all_errors.append(np.array([(parameters[0] -
                                     np.log(focal_initial)) * 8]))
        return np.concatenate(all_errors)

    lower = np.r_[np.log(width * 0.45), np.full(3 * (count - 1), -np.inf)]
    upper = np.r_[np.log(width * 3.0), np.full(3 * (count - 1), np.inf)]
    damping = 1e-2
    old_error = residual(x)
    old_cost = np.sum(np.sqrt(1 + old_error ** 2) - 1)
    for _ in range(30):
        jacobian = np.empty((len(old_error), len(x)))
        for column in range(len(x)):
            step = 1e-5 * max(1.0, abs(x[column]))
            perturbed = x.copy()
            perturbed[column] += step
            jacobian[:, column] = (residual(perturbed) - old_error) / step
        weights = (1 + old_error ** 2) ** (-0.25)
        weighted_jacobian = jacobian * weights[:, None]
        weighted_error = old_error * weights
        normal = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_error
        accepted = False
        for _ in range(8):
            matrix = normal + damping * np.diag(np.maximum(np.diag(normal), 1e-8))
            try:
                delta = np.linalg.solve(matrix, -gradient)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Camera optimization is singular") from exc
            trial = np.clip(x + delta, lower, upper)
            trial_error = residual(trial)
            trial_cost = np.sum(np.sqrt(1 + trial_error ** 2) - 1)
            if np.isfinite(trial_cost) and trial_cost < old_cost:
                x, old_error, old_cost = trial, trial_error, trial_cost
                damping = max(damping / 3, 1e-8)
                accepted = True
                break
            damping *= 10
        if not accepted or np.linalg.norm(delta) < 1e-6:
            break

    focal = float(np.exp(x[0]))
    rotations = [np.eye(3) for _ in images]
    for position, image_index in enumerate(variable_images):
        rotations[image_index] = cv2.Rodrigues(
            x[1 + position * 3:4 + position * 3])[0]
    # Level the horizon from the top row, then center it away from wrap-around.
    first_row = min(4, count)
    corrected = cv2.detail.waveCorrect(
        [r.astype(np.float32) for r in rotations[:first_row]],
        cv2.detail.WAVE_CORRECT_HORIZ)
    level = corrected[0] @ rotations[0].T
    rotations = [level @ rotation for rotation in rotations]
    forward = np.sum([rotation[:, 2] for rotation in rotations[:first_row]],
                     axis=0)
    angle = np.arctan2(forward[0], forward[2])
    recenter = cv2.Rodrigues(np.array([0.0, -angle, 0.0]))[0]
    rotations = [(recenter @ rotation).astype(np.float32)
                 for rotation in rotations]
    return focal, rotations


def _surface_layers(images, focal, rotations, projection):
    warper = cv2.PyRotationWarper(projection, float(focal))
    layers = []
    for image, rotation in zip(images, rotations):
        height, width = image.shape[:2]
        intrinsic = np.array([[focal, 0, (width - 1) / 2],
                              [0, focal, (height - 1) / 2],
                              [0, 0, 1]], np.float32)
        corner, warped = warper.warp(image, intrinsic, rotation,
                                     cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        _, valid = warper.warp(np.full((height, width), 255, np.uint8),
                               intrinsic, rotation, cv2.INTER_NEAREST,
                               cv2.BORDER_CONSTANT)
        layers.append((warped, valid, np.array(corner, np.int32)))
    return layers


def _surface_stitch(paths, count, projection, blend):
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        cv2.setRNGSeed(42)
        images = _read(paths, count)
        if count == 3:
            pairs = [(0, 1), (1, 2)]
        else:
            pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)
                     if abs(i // 4 - j // 4) <= 1
                     and abs(i % 4 - j % 4) <= 1]
        edges = _edges(images, pairs)
        if len(edges) < count - 1:
            raise ValueError("Too few reliable image-pair matches")
        focal, rotations = _camera_rotations(edges, images)
        return _compose(_surface_layers(images, focal, rotations, projection),
                        blend)
    finally:
        cv2.setNumThreads(old_threads)


def q2(img_paths):
    return _surface_stitch(img_paths, 3, "cylindrical", blend=False)


def q3(img_paths):
    return _surface_stitch(img_paths, 8, "cylindrical", blend=False)


def e1(img_paths):
    return _surface_stitch(img_paths, 8, "spherical", blend=False)


def e2(img_paths):
    return _surface_stitch(img_paths, 8, "spherical", blend=True)
