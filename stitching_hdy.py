"""
컴퓨터비전기초와응용 — 이미지 정합(Image Stitching) 과제

이 파일 하나만 구현해서 제출합니다.
  - 함수 이름, 인자, 반환 형식은 바꾸지 마세요. 조교가 run.py / grade.py로
    비공개 이미지에 대해 자동 실행합니다.
  - 각 함수는 최종 정합 결과 이미지 1장을 반환합니다.
    (cv2.imread와 같은 BGR uint8 numpy 배열)
  - 사용 가능 라이브러리: cv2, numpy, matplotlib, os, glob, math, pathlib 등 표준 라이브러리
    (cv2.Stitcher / cv2.Stitcher_create 사용 금지 — 특징점 검출, 매칭, 정합을 단계별로 구현)
  - 파일 내 다른 함수 정의는 자유입니다.
    단, 모든 코드는 이 파일 하나에 작성하세요. 직접 만든 다른 .py 파일을 import하면
    채점 시 import 오류로 실행되지 않습니다. (run.py도 이 경우 에러를 냅니다)
  - 구현하지 않은 문제(E1, E2 등)는 NotImplementedError를 그대로 두면 "미제출"로 처리됩니다.

입력 (5개 함수 모두 동일, 고정)
  - img_paths: 이미지 파일 경로(문자열)들이 담긴 리스트
      예) ["data/Q2/Q2_img_1.JPG", "data/Q2/Q2_img_2.JPG", "data/Q2/Q2_img_3.JPG"]
  - 파일명 번호 순서(1, 2, 3, ...)로 정렬되어 들어옵니다.
  - 입력 이미지는 img_paths로 받은 경로로만 읽으세요 (예: cv2.imread(p)).
    코드에 "data/Q2/..." 같은 경로를 직접 쓰지 마세요.

발표 자료용 단계별 결과(특징점, 매칭 등) 저장
  - 저장 경로는 자유롭게 써도 됩니다.
  - 단, 제출할 때는 저장 코드를 삭제하세요.
"""


import cv2
import numpy as np


def _q2_rootsift(descriptors):
    """Use the F06 L1-normalized square-root descriptor for Q2 matching."""
    if descriptors is None:
        return None
    values = np.asarray(descriptors, dtype=np.float32)
    totals = np.sum(np.abs(values), axis=1, keepdims=True)
    return np.sqrt(values / np.maximum(totals, 1e-12)).astype(np.float32)


def _q1_center_weight(height, width):
    """Brown-Lowe's centre-to-edge image weight on a planar canvas."""
    x = 1.0 - np.abs((np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0)
    y = 1.0 - np.abs((np.arange(height, dtype=np.float32) + 0.5) / height * 2.0 - 1.0)
    return y[:, None] * x[None, :]


def _q1_gaussian_pyramid(image, valid, levels):
    """Downsample without mixing the black area outside a warped image into it."""
    images = [image.astype(np.float32)]
    masks = [valid.astype(np.float32)]
    for _ in range(levels):
        next_mask = cv2.pyrDown(masks[-1])
        weighted = cv2.pyrDown(images[-1] * masks[-1][..., None])
        images.append(weighted / np.maximum(next_mask[..., None], 1e-6))
        masks.append(next_mask)
    return images, masks


def _q1_multiband_blend(image1, image2, mask1, mask2, weight1, weight2):
    """Blend Laplacian bands with Brown-Lowe's maximum-weight ownership map."""
    height, width = mask1.shape
    owner1 = (weight1 >= weight2).astype(np.float32)
    owner1[~mask1] = 0.0
    owner1[~mask2] = 1.0
    owner1[~(mask1 | mask2)] = 0.0

    levels = 0
    size = min(height, width)
    while size >= 32 and levels < 6:
        levels += 1
        size = (size + 1) // 2

    gaussian1, valid1 = _q1_gaussian_pyramid(image1, mask1, levels)
    gaussian2, valid2 = _q1_gaussian_pyramid(image2, mask2, levels)
    owner = [owner1]
    for _ in range(levels):
        owner.append(cv2.pyrDown(owner[-1]))

    blended_bands = []
    for level in range(levels):
        target_size = (gaussian1[level].shape[1], gaussian1[level].shape[0])
        band1 = gaussian1[level] - cv2.pyrUp(gaussian1[level + 1], dstsize=target_size)
        band2 = gaussian2[level] - cv2.pyrUp(gaussian2[level + 1], dstsize=target_size)
        alpha = owner[level].copy()
        alpha[valid1[level] < 1e-3] = 0.0
        alpha[valid2[level] < 1e-3] = 1.0
        blended_bands.append(band1 * alpha[..., None] + band2 * (1.0 - alpha[..., None]))

    alpha = owner[-1].copy()
    alpha[valid1[-1] < 1e-3] = 0.0
    alpha[valid2[-1] < 1e-3] = 1.0
    result = gaussian1[-1] * alpha[..., None] + gaussian2[-1] * (1.0 - alpha[..., None])
    for level in range(levels - 1, -1, -1):
        target_size = (blended_bands[level].shape[1], blended_bands[level].shape[0])
        result = cv2.pyrUp(result, dstsize=target_size) + blended_bands[level]

    result[~(mask1 | mask2)] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def _q1_canvas_geometry(image1, image2, homography):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    corners1 = np.float32([[0, 0], [width1 - 1, 0], [width1 - 1, height1 - 1], [0, height1 - 1]])
    corners2 = np.float32([[0, 0], [width2 - 1, 0], [width2 - 1, height2 - 1], [0, height2 - 1]])
    projected2 = cv2.perspectiveTransform(corners2.reshape(-1, 1, 2), homography).reshape(-1, 2)
    corners = np.vstack((corners1, projected2))
    if not np.isfinite(corners).all():
        raise ValueError("호모그래피가 유효하지 않은 캔버스 좌표를 만들었습니다")

    minimum = np.floor(corners.min(axis=0)).astype(np.int64)
    maximum = np.ceil(corners.max(axis=0)).astype(np.int64) + 1
    out_width, out_height = (maximum - minimum).tolist()
    input_area = width1 * height1 + width2 * height2
    if out_width <= 0 or out_height <= 0 or out_width * out_height > 20 * input_area:
        raise ValueError("정합 캔버스 크기가 비정상적입니다")
    translation = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    return translation, (int(out_width), int(out_height))


def q1(img_paths):
    """Q1. planar(평면) 환경에서 2장 정합.

    Brown-Lowe의 SIFT 특징 매칭과 RANSAC 호모그래피를 두 장의 평면
    정합에 적용하고, 겹침 영역은 멀티밴드 방식으로 합성한다.

    Returns: 정합 결과 이미지 (ndarray)
    """
    if len(img_paths) != 2:
        raise ValueError(f"Q1은 이미지 2장이 필요합니다 (입력: {len(img_paths)}장)")

    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in img_paths]
    for path, image in zip(img_paths, images):
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    image1, image2 = images

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
    if descriptors1 is None or descriptors2 is None or len(descriptors1) < 2 or len(descriptors2) < 2:
        raise ValueError("호모그래피 추정에 필요한 SIFT 특징점이 부족합니다")

    # FLANN의 k-d tree로 후보를 찾고, 모호한 대응은 Lowe의 최근접 거리 비율로 거른다.
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    candidates = matcher.knnMatch(descriptors2, descriptors1, k=2)
    matches = [pair[0] for pair in candidates if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 4:
        raise ValueError(f"호모그래피 추정에 필요한 특징점 대응이 부족합니다 ({len(matches)}개)")

    source = np.float32([keypoints2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(
        source, target, cv2.RANSAC, 3.0, maxIters=500, confidence=0.995
    )
    if homography is None or inliers is None or int(inliers.sum()) < 8:
        raise ValueError("RANSAC으로 신뢰할 만한 평면 호모그래피를 찾지 못했습니다")

    translation, canvas_size = _q1_canvas_geometry(image1, image2, homography)
    # Large private inputs are rendered at a bounded resolution to keep the
    # multi-band pyramids within a predictable amount of memory.
    max_canvas_pixels = 8_000_000
    if canvas_size[0] * canvas_size[1] > max_canvas_pixels:
        scale = 0.98 * np.sqrt(max_canvas_pixels / (canvas_size[0] * canvas_size[1]))
        height1, width1 = image1.shape[:2]
        height2, width2 = image2.shape[:2]
        size1 = (max(1, round(width1 * scale)), max(1, round(height1 * scale)))
        size2 = (max(1, round(width2 * scale)), max(1, round(height2 * scale)))
        image1 = cv2.resize(image1, size1, interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(image2, size2, interpolation=cv2.INTER_AREA)
        scale1 = np.diag([size1[0] / width1, size1[1] / height1, 1.0])
        scale2 = np.diag([size2[0] / width2, size2[1] / height2, 1.0])
        homography = scale1 @ homography @ np.linalg.inv(scale2)
        translation, canvas_size = _q1_canvas_geometry(image1, image2, homography)

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    transform1, transform2 = translation, translation @ homography

    warped1 = cv2.warpPerspective(image1, transform1, canvas_size, borderMode=cv2.BORDER_REPLICATE)
    warped2 = cv2.warpPerspective(image2, transform2, canvas_size, borderMode=cv2.BORDER_REPLICATE)
    mask1 = cv2.warpPerspective(np.ones((height1, width1), np.uint8), transform1, canvas_size,
                                flags=cv2.INTER_NEAREST) > 0
    mask2 = cv2.warpPerspective(np.ones((height2, width2), np.uint8), transform2, canvas_size,
                                flags=cv2.INTER_NEAREST) > 0
    weight1 = cv2.warpPerspective(_q1_center_weight(height1, width1), transform1, canvas_size)
    weight2 = cv2.warpPerspective(_q1_center_weight(height2, width2), transform2, canvas_size)
    return _q1_multiband_blend(warped1, warped2, mask1, mask2, weight1, weight2)


def _q2_cylindrical_project(image, focal_length=None):
    """Project a BGR image onto a cylinder; return image and valid-pixel mask."""
    height, width = image.shape[:2]
    # The supplied Q2 photos have a 27 mm full-frame-equivalent lens (f=0.75w).
    # Use the same image-size heuristic when camera calibration is unavailable.
    focal = float(focal_length) if focal_length is not None else 0.75 * width
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Cylindrical focal length must be positive")
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    theta = (np.arange(width, dtype=np.float32) - centre_x) / focal
    vertical = np.arange(height, dtype=np.float32) - centre_y
    source_x = np.broadcast_to(focal * np.tan(theta)[None, :] + centre_x,
                               (height, width)).astype(np.float32)
    source_y = (vertical[:, None] / np.cos(theta)[None, :] + centre_y).astype(np.float32)
    valid = ((source_x >= 0) & (source_x <= width - 1) &
             (source_y >= 0) & (source_y <= height - 1))
    projected = cv2.remap(image, source_x, source_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    return projected, valid


def _q3_spherical_project(image, focal_length=None):
    """Project one BGR view to longitude/latitude coordinates with a valid mask."""
    height, width = image.shape[:2]
    focal = float(focal_length) if focal_length is not None else float(width)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Spherical focal length must be positive")
    centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    theta = (np.arange(width, dtype=np.float32) - centre_x) / focal
    phi = (np.arange(height, dtype=np.float32) - centre_y) / focal
    source_x = np.broadcast_to(centre_x + focal * np.tan(theta)[None, :],
                               (height, width)).astype(np.float32)
    source_y = (centre_y + focal * np.tan(phi[:, None]) /
                np.cos(theta)[None, :]).astype(np.float32)
    valid = ((source_x >= 0) & (source_x <= width - 1) &
             (source_y >= 0) & (source_y <= height - 1))
    projected = cv2.remap(image, source_x, source_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    return projected, valid


def _q2_pair_homography(source_features, target_features, matcher, return_details=False):
    """Estimate a Q2 homography from RootSIFT matches with MAGSAC."""
    source_keypoints, source_descriptors = source_features
    target_keypoints, target_descriptors = target_features
    if source_descriptors is None or target_descriptors is None:
        raise ValueError("Not enough SIFT features for Q2 alignment")
    candidates = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [pair[0] for pair in candidates
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 12:
        raise ValueError("Not enough reliable Q2 feature matches")
    source = np.float32([source_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([target_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(
        source, target, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.995
    )
    if (homography is None or inliers is None or int(inliers.sum()) < 12 or
            not np.isfinite(homography).all()):
        raise ValueError("RANSAC could not align the adjacent Q2 images")
    if return_details:
        return homography, matches, inliers.ravel().astype(bool)
    return homography


def _q2_sift_features(images, masks):
    """Detect the same masked SIFT features used by q2 for each image."""
    sift = cv2.SIFT_create(nfeatures=6000)
    features = []
    for image, mask in zip(images, masks):
        interior = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8))
        features.append(sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), interior))
    return features


def _q2_f06_features(images, masks):
    """Keep SIFT keypoints but apply RootSIFT only to Q2 descriptors."""
    return [(keypoints, _q2_rootsift(descriptors))
            for keypoints, descriptors in _q2_sift_features(images, masks)]


def _q2_align(img_paths):
    """Return cylindrical images, masks, and transforms into image 2's frame.

    Diagnostic and evaluation code can call this to use the final stitch's
    exact projection and pre-blend alignment.
    """
    if len(img_paths) != 3:
        raise ValueError(f"Q2 requires three images (received {len(img_paths)})")
    cv2.setRNGSeed(42)
    originals = [cv2.imread(path, cv2.IMREAD_COLOR) for path in img_paths]
    for path, image in zip(img_paths, originals):
        if image is None:
            raise ValueError(f"Could not read image: {path}")
    projected_and_masks = [_q2_cylindrical_project(image) for image in originals]
    images = [item[0] for item in projected_and_masks]
    masks = [item[1] for item in projected_and_masks]
    features = _q2_f06_features(images, masks)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    transforms = [
        _q2_pair_homography(features[0], features[1], matcher),
        np.eye(3, dtype=np.float64),
        _q2_pair_homography(features[2], features[1], matcher),
    ]
    return images, masks, transforms


def _q2_canvas_geometry(masks, transforms):
    """Fit the three valid cylindrical footprints into one canvas."""
    footprints = []
    for mask, transform in zip(masks, transforms):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Cylindrical projection has no valid pixels")
        boundary = max(contours, key=cv2.contourArea).astype(np.float32)
        footprints.append(cv2.perspectiveTransform(boundary, transform).reshape(-1, 2))
    points = np.vstack(footprints)
    if not np.isfinite(points).all():
        raise ValueError("Invalid Q2 panorama coordinates")
    minimum = np.floor(points.min(axis=0)).astype(np.int64)
    maximum = np.ceil(points.max(axis=0)).astype(np.int64) + 1
    out_width, out_height = (maximum - minimum).tolist()
    input_area = sum(mask.size for mask in masks)
    if out_width <= 0 or out_height <= 0 or out_width * out_height > 20 * input_area:
        raise ValueError("Q2 alignment produced an unreasonable canvas")
    translation = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]],
                           dtype=np.float64)
    return translation, (int(out_width), int(out_height))


def _q2_multiband_blend(images, masks, weights):
    """Blend three aligned images using Brown-Lowe centre-weight ownership."""
    height, width = masks[0].shape
    scores = np.stack([np.where(mask, weight, -1.0)
                       for mask, weight in zip(masks, weights)])
    chosen = np.argmax(scores, axis=0)
    owners = [((chosen == index) & mask).astype(np.float32)
              for index, mask in enumerate(masks)]
    levels, size = 0, min(height, width)
    while size >= 32 and levels < 6:
        levels += 1
        size = (size + 1) // 2
    pyramids = [_q1_gaussian_pyramid(image, mask, levels)
                for image, mask in zip(images, masks)]
    owner_pyramids = []
    for owner in owners:
        pyramid = [owner]
        for _ in range(levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        owner_pyramids.append(pyramid)
    bands = []
    for level in range(levels):
        target_size = (pyramids[0][0][level].shape[1], pyramids[0][0][level].shape[0])
        alphas = [owner_pyramids[i][level] * (pyramids[i][1][level] > 1e-3)
                  for i in range(3)]
        denominator = np.maximum(sum(alphas), 1e-6)
        blended = np.zeros((*alphas[0].shape, 3), np.float32)
        for i in range(3):
            laplacian = (pyramids[i][0][level] -
                         cv2.pyrUp(pyramids[i][0][level + 1], dstsize=target_size))
            blended += laplacian * (alphas[i] / denominator)[..., None]
        bands.append(blended)
    alphas = [owner_pyramids[i][-1] * (pyramids[i][1][-1] > 1e-3)
              for i in range(3)]
    denominator = np.maximum(sum(alphas), 1e-6)
    result = np.zeros((*alphas[0].shape, 3), np.float32)
    for i in range(3):
        result += pyramids[i][0][-1] * (alphas[i] / denominator)[..., None]
    for level in range(levels - 1, -1, -1):
        target_size = (bands[level].shape[1], bands[level].shape[0])
        result = cv2.pyrUp(result, dstsize=target_size) + bands[level]
    result[~np.logical_or.reduce(masks)] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def _q2_crop_valid_rectangle(panorama, valid):
    """Remove cylindrical black padding using only the warped validity masks."""
    height, width = valid.shape
    occupied = np.flatnonzero(valid.any(axis=0))
    if not occupied.size:
        return panorama
    left, right = int(occupied[0]), int(occupied[-1])
    first = np.argmax(valid, axis=0)
    last = height - 1 - np.argmax(valid[::-1], axis=0)
    occupied_columns = valid.any(axis=0)
    invalid_integral = cv2.integral((~valid).astype(np.uint8))
    best = None
    best_area = 0
    step = max(1, (right - left + 1) // 500)
    for margin in range(0, (right - left + 1) // 10 + 1, step):
        x0, x1 = left + margin, right - margin + 1
        if x1 <= x0 or not occupied_columns[x0:x1].all():
            continue
        y0, y1 = int(first[x0:x1].max()), int(last[x0:x1].min()) + 1
        if y1 <= y0:
            continue
        invalid_count = (invalid_integral[y1, x1] - invalid_integral[y0, x1] -
                         invalid_integral[y1, x0] + invalid_integral[y0, x0])
        area = (x1 - x0) * (y1 - y0)
        if invalid_count == 0 and area > best_area:
            best, best_area = (x0, y0, x1, y1), area
    if best is None:
        return panorama
    x0, y0, x1, y1 = best
    return panorama[y0:y1, x0:x1]


def q2(img_paths):
    """Q2: cylindrical SIFT/RootSIFT matching, MAGSAC and multiband blending."""
    images, masks, transforms = _q2_align(img_paths)
    translation, canvas_size = _q2_canvas_geometry(masks, transforms)
    max_canvas_pixels = 8_000_000
    if canvas_size[0] * canvas_size[1] > max_canvas_pixels:
        scale = 0.98 * np.sqrt(max_canvas_pixels / (canvas_size[0] * canvas_size[1]))
        resized_images, resized_masks, scaled_transforms = [], [], []
        for image, mask, transform in zip(images, masks, transforms):
            height, width = image.shape[:2]
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized_images.append(cv2.resize(image, new_size, interpolation=cv2.INTER_AREA))
            resized_masks.append(cv2.resize(mask.astype(np.uint8), new_size,
                                            interpolation=cv2.INTER_NEAREST) > 0)
            image_scale = np.diag([new_size[0] / width, new_size[1] / height, 1.0])
            canvas_scale = np.diag([scale, scale, 1.0])
            scaled_transforms.append(canvas_scale @ transform @ np.linalg.inv(image_scale))
        images, masks, transforms = resized_images, resized_masks, scaled_transforms
        translation, canvas_size = _q2_canvas_geometry(masks, transforms)
    warped_images, warped_masks, warped_weights = [], [], []
    for image, mask, transform in zip(images, masks, transforms):
        height, width = image.shape[:2]
        matrix = translation @ transform
        warped_images.append(cv2.warpPerspective(image, matrix, canvas_size,
                                                 borderMode=cv2.BORDER_REPLICATE))
        warped_masks.append(cv2.warpPerspective(mask.astype(np.uint8), matrix, canvas_size,
                                                flags=cv2.INTER_NEAREST) > 0)
        centre_weight = _q1_center_weight(height, width) * mask
        warped_weights.append(cv2.warpPerspective(centre_weight, matrix, canvas_size))
    panorama = _q2_multiband_blend(warped_images, warped_masks, warped_weights)
    valid = np.logical_or.reduce(warped_masks)
    return _q2_crop_valid_rectangle(panorama, valid)


def _q3_pair_similarity(features, source_index, target_index, matcher, scales):
    """Lowe-ratio matches and a bounded RANSAC similarity for one image pair."""
    source_keypoints, source_descriptors = features[source_index]
    target_keypoints, target_descriptors = features[target_index]
    if source_descriptors is None or target_descriptors is None or len(target_descriptors) < 2:
        return None
    candidates = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [pair[0] for pair in candidates
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 12:
        return None
    source = np.float32([source_keypoints[m.queryIdx].pt for m in matches])
    target = np.float32([target_keypoints[m.trainIdx].pt for m in matches])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0,
        maxIters=2000, confidence=0.995, refineIters=10
    )
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return None
    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < 12:
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = abs(float(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if not 0.6 <= scale <= 1.6 or angle > np.deg2rad(45):
        return None
    return {
        "source": source_index, "target": target_index,
        "matches": matches, "inliers": inliers,
        "homography": np.vstack((matrix, [0.0, 0.0, 1.0])),
        "src_points": source[inliers], "dst_points": target[inliers],
        "scale_source": scales[source_index], "scale_target": scales[target_index],
        "weight": 1.0,
    }


def _q3_top_order(candidate, width):
    """Use verified horizontal displacement to detect a misplaced top-row photo."""
    scores = [0] * 4
    for source in range(4):
        for target in range(source + 1, 4):
            edge = candidate(source, target)
            if edge is None or int(edge["inliers"].sum()) < 20:
                continue
            displacement = float(np.median(edge["src_points"][:, 0] -
                                           edge["dst_points"][:, 0]))
            if abs(displacement) < 0.08 * width:
                continue
            if displacement > 0:  # A left image sees the overlap on its right.
                scores[source] += 1
                scores[target] -= 1
            else:
                scores[source] -= 1
                scores[target] += 1
    return sorted(range(4), key=lambda index: (-scores[index], index))


def _q3_global_similarity(edges):
    """Align all eight cylinders jointly from pairwise RANSAC inlier points.

    A per-image similarity has four unknowns. Holding image 6 fixed removes
    the gauge freedom; the remaining 28 unknowns form one linear least-squares
    system. Giving each edge a similar total weight prevents a long horizontal
    edge from completely overwhelming the short top-to-bottom overlap.
    """
    anchor = 5
    indices = {image: index for index, image in
               enumerate(i for i in range(8) if i != anchor)}
    blocks, targets = [], []
    rng = np.random.default_rng(42)
    for edge in edges:
        source_index, target_index = edge["source"], edge["target"]
        source, target = edge["src_points"], edge["dst_points"]
        if len(source) > 400:
            selected = np.sort(rng.choice(len(source), size=400, replace=False))
            source, target = source[selected], target[selected]
        count = len(source)
        block = np.zeros((2 * count, 28), np.float64)
        rhs = np.zeros(2 * count, np.float64)
        x, y = source[:, 0], source[:, 1]
        u, v = target[:, 0], target[:, 1]
        if source_index == anchor:
            rhs[0::2] -= x
            rhs[1::2] -= y
        else:
            offset = 4 * indices[source_index]
            block[0::2, offset] = x
            block[0::2, offset + 1] = -y
            block[0::2, offset + 2] = 1
            block[1::2, offset] = y
            block[1::2, offset + 1] = x
            block[1::2, offset + 3] = 1
        if target_index == anchor:
            rhs[0::2] += u
            rhs[1::2] += v
        else:
            offset = 4 * indices[target_index]
            block[0::2, offset] -= u
            block[0::2, offset + 1] += v
            block[0::2, offset + 2] -= 1
            block[1::2, offset] -= v
            block[1::2, offset + 1] -= u
            block[1::2, offset + 3] -= 1
        weight = edge["weight"] / np.sqrt(count)
        blocks.append(block * weight)
        targets.append(rhs * weight)
    solution, _, rank, _ = np.linalg.lstsq(np.vstack(blocks),
                                           np.concatenate(targets), rcond=None)
    if rank != 28:
        raise ValueError("Q3 대응점으로 전역 정합을 결정할 수 없습니다")
    transforms = []
    for image in range(8):
        if image == anchor:
            transforms.append(np.eye(3, dtype=np.float64))
            continue
        a, b, tx, ty = solution[4 * indices[image]:4 * indices[image] + 4]
        scale = np.hypot(a, b)
        if not np.isfinite([a, b, tx, ty]).all() or not 0.5 <= scale <= 2.0:
            raise ValueError("Q3 전역 변환의 배율이 비정상적입니다")
        transforms.append(np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]],
                                   dtype=np.float64))
    return transforms


def _q3_align(img_paths, return_details=False, projection="cylindrical"):
    """Return projected images, validity masks, and common-frame transforms."""
    if projection not in ("cylindrical", "spherical"):
        raise ValueError(f"Unsupported Q3 projection: {projection}")
    if len(img_paths) != 8:
        raise ValueError(f"Q3은 이미지 8장이 필요합니다 (입력: {len(img_paths)}장)")
    cv2.setRNGSeed(42)
    originals = [cv2.imread(path, cv2.IMREAD_COLOR) for path in img_paths]
    for path, image in zip(img_paths, originals):
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    # The common scale bounds both SIFT work and the eight-image warp memory.
    longest = max(max(image.shape[:2]) for image in originals)
    common_scale = min(1.0, 1008.0 / longest)
    resized, scales = [], []
    for image in originals:
        height, width = image.shape[:2]
        size = (max(1, round(width * common_scale)),
                max(1, round(height * common_scale)))
        resized.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                       if common_scale < 1.0 else image)
        scales.append((size[0] / width, size[1] / height))
    del originals
    # f = image width is an uncalibrated Q3 heuristic, distinct from Q2's f=.75w.
    projector = (_q2_cylindrical_project if projection == "cylindrical"
                 else _q3_spherical_project)
    projected = [projector(image, float(image.shape[1])) for image in resized]
    images = [item[0] for item in projected]
    masks = [item[1] for item in projected]
    del projected, resized
    features = _q2_sift_features(images, masks)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    cache = {}

    def candidate(source, target):
        key = (source, target)
        if key not in cache:
            cache[key] = _q3_pair_similarity(features, source, target,
                                             matcher, scales)
        return cache[key]

    top = _q3_top_order(candidate, images[0].shape[1])
    edge_details = []
    horizontal = ([(top[i], top[i + 1]) for i in range(3)] +
                  [(i, i + 1) for i in range(4, 7)])
    for source, target in horizontal:
        edge = candidate(source, target)
        if edge is None or int(edge["inliers"].sum()) < 15:
            raise ValueError(f"Q3 가로 사진 {source + 1}–{target + 1}의 검증된 대응점이 부족합니다")
        edge_details.append(edge)
    for row in (top, list(range(4, 8))):
        for offset in range(2):
            edge = candidate(row[offset], row[offset + 2])
            if edge is not None and int(edge["inliers"].sum()) >= 30:
                edge["weight"] = 0.7
                edge_details.append(edge)
    vertical_count = 0
    for source in range(4):
        for target in range(4, 8):
            edge = candidate(source, target)
            if edge is None or int(edge["inliers"].sum()) < 20:
                continue
            if (np.median(edge["src_points"][:, 1]) < 0.6 * images[source].shape[0] or
                    np.median(edge["dst_points"][:, 1]) > 0.4 * images[target].shape[0]):
                continue
            edge["weight"] = 2.0
            edge_details.append(edge)
            vertical_count += 1
    if vertical_count == 0:
        raise ValueError("Q3 위아래 줄 사이에 검증된 공통 영역이 없습니다")
    transforms = _q3_global_similarity(edge_details)
    if return_details:
        return images, masks, transforms, features, edge_details
    return images, masks, transforms


def _q3_multiband_blend(images, masks, transforms, translation, canvas_size,
                        max_levels=6, seam_rows=((4, 5, 6, 7),)):
    """Find low-cost overlap seams, then blend bands without keeping eight pyramids."""
    width, height = canvas_size
    matrices = [translation @ transform for transform in transforms]
    owner = np.full((height, width), 255, np.uint8)
    best_weight = np.zeros((height, width), np.float32)
    for index, (image, mask, matrix) in enumerate(zip(images, masks, matrices)):
        ih, iw = image.shape[:2]
        valid = cv2.warpPerspective(mask.astype(np.uint8), matrix, canvas_size,
                                    flags=cv2.INTER_NEAREST) > 0
        weight = cv2.warpPerspective(_q1_center_weight(ih, iw) * mask,
                                     matrix, canvas_size)
        choose = valid & ((weight > best_weight) | (owner == 255))
        owner[choose] = index
        best_weight[choose] = weight[choose]
    del best_weight
    # Preserve the original top-row and top-to-bottom ownership. Their small
    # common regions have weak image correspondence; an unconstrained seam can
    # carry dark tree pixels into the lower white building. Optimise the
    # bottom row, whose adjacent views have stronger overlap on the road.
    if hasattr(cv2, "detail_DpSeamFinder"):
        small_size = (max(1, round(width / 4)), max(1, round(height / 4)))
        reduction = np.diag([small_size[0] / width,
                             small_size[1] / height, 1.0])
        for row in seam_rows:
            if len(row) < 2:
                continue
            row_region = np.isin(owner, row)
            small_images, small_masks = [], []
            for index in row:
                small_matrix = reduction @ matrices[index]
                small_images.append(cv2.warpPerspective(images[index], small_matrix,
                                                         small_size).astype(np.float32))
                small_masks.append(cv2.warpPerspective(masks[index].astype(np.uint8) * 255,
                                                        small_matrix, small_size,
                                                        flags=cv2.INTER_NEAREST))
            seam_masks = cv2.detail_DpSeamFinder("COLOR_GRAD").find(
                small_images, [(0, 0)] * len(row), small_masks
            )
            seam_masks = [mask.get() if isinstance(mask, cv2.UMat) else mask
                          for mask in seam_masks]
            local_owner = np.argmax(np.stack([mask > 0 for mask in seam_masks]),
                                    axis=0).astype(np.uint8)
            seam_owner = cv2.resize(local_owner, canvas_size,
                                    interpolation=cv2.INTER_NEAREST)
            for local_index, index in enumerate(row):
                valid = cv2.warpPerspective(masks[index].astype(np.uint8),
                                            matrices[index], canvas_size,
                                            flags=cv2.INTER_NEAREST) > 0
                owner[row_region & (seam_owner == local_index) & valid] = index
    occupied = owner != 255
    levels, size = 0, min(height, width)
    while size >= 32 and levels < max_levels:
        levels += 1
        size = (size + 1) // 2
    shapes = [(height, width)]
    for _ in range(levels):
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    bands = [np.zeros((*shape, 3), np.float32) for shape in shapes]
    denominators = [np.zeros(shape, np.float32) for shape in shapes]
    for index, (image, mask, matrix) in enumerate(zip(images, masks, matrices)):
        warped = cv2.warpPerspective(image, matrix, canvas_size,
                                     borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(mask.astype(np.uint8), matrix, canvas_size,
                                    flags=cv2.INTER_NEAREST) > 0
        gaussian, valid_pyramid = _q1_gaussian_pyramid(warped, valid, levels)
        ownership = [(owner == index).astype(np.float32)]
        for _ in range(levels):
            ownership.append(cv2.pyrDown(ownership[-1]))
        for level in range(levels + 1):
            alpha = ownership[level] * (valid_pyramid[level] > 1e-3)
            denominators[level] += alpha
            if level == levels:
                band = gaussian[level]
            else:
                size = (shapes[level][1], shapes[level][0])
                band = gaussian[level] - cv2.pyrUp(gaussian[level + 1],
                                                   dstsize=size)
            bands[level] += band * alpha[..., None]
    panorama = bands[-1] / np.maximum(denominators[-1][..., None], 1e-6)
    for level in range(levels - 1, -1, -1):
        size = (shapes[level][1], shapes[level][0])
        panorama = (cv2.pyrUp(panorama, dstsize=size) +
                    bands[level] / np.maximum(denominators[level][..., None], 1e-6))
    panorama[~occupied] = 0
    return np.clip(panorama, 0, 255).astype(np.uint8)


def q3(img_paths):
    """Q3: cylindrical Brown-Lowe features with a local APAP mesh warp."""
    return _hdy_apap_panorama(img_paths, projection="cylindrical", blend=True)


def _e1e2_panorama(img_paths, blend):
    """Use the same spherical APAP geometry with hard or multiband composition."""
    return _hdy_apap_panorama(img_paths, projection="spherical", blend=blend)


def e1(img_paths):
    """E1 (추가점수). Q3의 8장을 spherical(구면) 투영 후 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    return _e1e2_panorama(img_paths, blend=False)


def e2(img_paths):
    """E2 (추가점수). E1 결과에 blending을 적용해 자연스럽게 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    return _e1e2_panorama(img_paths, blend=True)


# This is a local Moving DLT (APAP) port over the Brown-Lowe global frame.
# It is self-contained: no model weights, cached photos, or experiment modules.
# It does not claim to reproduce the authors' MATLAB compositor exactly.
class _HdyMovingDLT:
    def __init__(self, source, target, sigma_px=60.0, gamma=0.1):
        self.source = np.asarray(source, np.float64).reshape(-1, 2)
        self.target = np.asarray(target, np.float64).reshape(-1, 2)
        if self.source.shape != self.target.shape or len(self.source) < 12:
            raise ValueError("APAP needs at least 12 paired RANSAC inliers")
        self.sigma_px = float(sigma_px)
        self.gamma = float(gamma)
        sn, self.ts = self._normalize(self.source)
        tn, tt = self._normalize(self.target)
        self.inv_tt = np.linalg.inv(tt)
        n = len(sn)
        a = np.zeros((n, 2, 9), np.float64)
        x, y, u, v = sn[:, 0], sn[:, 1], tn[:, 0], tn[:, 1]
        a[:, 0, 0:3] = np.column_stack((-x, -y, -np.ones(n)))
        a[:, 0, 6:9] = np.column_stack((u*x, u*y, u))
        a[:, 1, 3:6] = np.column_stack((-x, -y, -np.ones(n)))
        a[:, 1, 6:9] = np.column_stack((v*x, v*y, v))
        self.aa = np.einsum("nij,nik->njk", a, a)
        self.base = (self.gamma ** 2) * self.aa.sum(axis=0)

    @staticmethod
    def _normalize(points):
        centre = points.mean(axis=0)
        radius = np.linalg.norm(points-centre, axis=1).mean()
        if radius < 1e-8:
            raise ValueError("Degenerate APAP correspondences")
        scale = np.sqrt(2.0) / radius
        transform = np.array([[scale, 0, -scale*centre[0]],
                              [0, scale, -scale*centre[1]],
                              [0, 0, 1]], np.float64)
        return (points-centre)*scale, transform

    def transform(self, points):
        query = np.asarray(points, np.float64).reshape(-1, 2)
        answer = np.empty_like(query)
        for index, point in enumerate(query):
            delta = self.source-point
            distance2 = np.einsum("ij,ij->i", delta, delta)
            weights = np.maximum(np.exp(-distance2/self.sigma_px**2),
                                 self.gamma)
            changed = weights > self.gamma*(1+1e-12)
            matrix = self.base.copy()
            if np.any(changed):
                excess = weights[changed]**2-self.gamma**2
                matrix += np.einsum("n,njk->jk", excess, self.aa[changed])
            _, vectors = np.linalg.eigh(matrix)
            h = self.inv_tt @ vectors[:, 0].reshape(3, 3) @ self.ts
            if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
                raise ValueError("Invalid local APAP homography")
            homogeneous = h @ np.array([point[0], point[1], 1.0])
            answer[index] = homogeneous[:2]/homogeneous[2]
        return answer


def _hdy_apap_project(matrix, points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    hom = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return hom[:, :2]/hom[:, 2, None]


def _hdy_apap_mesh(source, target, shape_hw):
    height, width = shape_hw
    gx = np.linspace(0, width-1, 76)
    gy = np.linspace(0, height-1, 101)
    src = np.stack(np.meshgrid(gx, gy), axis=-1)
    mapped = _HdyMovingDLT(source, target).transform(src.reshape(-1, 2))
    mapped = mapped.reshape(src.shape)
    if not np.isfinite(mapped).all():
        raise ValueError("APAP mesh contains nonfinite vertices")
    return src, mapped


def _hdy_apap_mesh_maps(src_grid, dst_grid, output_shape, to_frame):
    height, width = output_shape
    map_x = np.full((height, width), -1.0, np.float32)
    map_y = np.full((height, width), -1.0, np.float32)
    assigned = np.zeros((height, width), np.uint8)
    dst = _hdy_apap_project(to_frame, dst_grid.reshape(-1, 2)).reshape(dst_grid.shape)
    rows, columns = src_grid.shape[:2]
    for iy in range(rows-1):
        for ix in range(columns-1):
            for ids in (((iy, ix), (iy, ix+1), (iy+1, ix)),
                        ((iy+1, ix+1), (iy+1, ix), (iy, ix+1))):
                src3 = np.float32([src_grid[a, b] for a, b in ids])
                dst3 = np.float32([dst[a, b] for a, b in ids])
                area2 = np.linalg.det(np.column_stack((dst3[1]-dst3[0],
                                                       dst3[2]-dst3[0])))
                if not np.isfinite(dst3).all() or abs(area2) < 1e-5:
                    continue
                x0 = max(0, int(np.floor(dst3[:, 0].min())))
                x1 = min(width, int(np.ceil(dst3[:, 0].max()))+1)
                y0 = max(0, int(np.floor(dst3[:, 1].min())))
                y1 = min(height, int(np.ceil(dst3[:, 1].max()))+1)
                if x1 <= x0 or y1 <= y0:
                    continue
                triangle = np.zeros((y1-y0, x1-x0), np.uint8)
                polygon = np.rint(dst3-np.array([x0, y0], np.float32)).astype(np.int32)
                cv2.fillConvexPoly(triangle, polygon, 1)
                yy, xx = np.nonzero(triangle)
                if len(xx) == 0:
                    continue
                px = xx.astype(np.float64)+x0
                py = yy.astype(np.float64)+y0
                affine = cv2.getAffineTransform(dst3, src3)
                map_x[py.astype(int), px.astype(int)] = (
                    affine[0, 0]*px+affine[0, 1]*py+affine[0, 2]).astype(np.float32)
                map_y[py.astype(int), px.astype(int)] = (
                    affine[1, 0]*px+affine[1, 1]*py+affine[1, 2]).astype(np.float32)
                assigned[py.astype(int), px.astype(int)] = 1
    return map_x, map_y, assigned.astype(bool)


def _hdy_apap_render(image, source_valid, src_grid, dst_grid,
                     output_shape, to_frame):
    mx, my, assigned = _hdy_apap_mesh_maps(src_grid, dst_grid,
                                           output_shape, to_frame)
    warped = cv2.remap(image, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    fraction = cv2.remap(source_valid.astype(np.float32), mx, my,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = assigned & (fraction > 0.999)
    warped[~valid] = 0
    return warped, valid


def _hdy_apap_panorama(img_paths, projection, blend, return_details=False):
    """Fit APAP from live RANSAC inliers and compose eight registered layers.

    `return_details=True` exposes the exact preblend layers and masks used in
    the output, for a reproducible overlap SSIM measurement. The public q3,
    e1, and e2 functions still return one BGR uint8 image each.
    """
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        return _hdy_apap_panorama_impl(img_paths, projection, blend,
                                       return_details)
    finally:
        cv2.setNumThreads(previous_threads)


def _hdy_apap_panorama_impl(img_paths, projection, blend, return_details):
    images, masks, transforms, features, edges = _q3_align(
        img_paths, return_details=True, projection=projection)
    if len(images) != 8 or len(edges) < 7:
        raise ValueError("APAP requires a connected eight-image RANSAC graph")
    sources = [[] for _ in images]
    targets = [[] for _ in images]
    for edge in edges:
        i, j = edge["source"], edge["target"]
        source = np.asarray(edge["src_points"], np.float64)
        target = np.asarray(edge["dst_points"], np.float64)
        consensus = 0.5*(_hdy_apap_project(transforms[i], source)+
                         _hdy_apap_project(transforms[j], target))
        sources[i].append(source)
        sources[j].append(target)
        targets[i].append(consensus)
        targets[j].append(consensus)
    grids = []
    for i, image in enumerate(images):
        if not sources[i]:
            raise ValueError(f"Image {i+1} has no APAP correspondences")
        source = np.concatenate(sources[i])
        target = np.concatenate(targets[i])
        if len(source) < 12:
            raise ValueError(f"Image {i+1} has too few APAP inliers")
        height, width = image.shape[:2]
        px, py = np.meshgrid(np.linspace(0, width-1, 5),
                             np.linspace(0, height-1, 5))
        prior_source = np.stack((px, py), axis=-1).reshape(-1, 2)
        prior_target = _hdy_apap_project(transforms[i], prior_source)
        repeats = max(1, round(0.5*len(source)/len(prior_source)))
        fit_source = np.concatenate((source,
                                     np.repeat(prior_source, repeats, axis=0)))
        fit_target = np.concatenate((target,
                                     np.repeat(prior_target, repeats, axis=0)))
        grids.append(_hdy_apap_mesh(fit_source, fit_target, image.shape[:2]))
    brown_translation, brown_size = _q2_canvas_geometry(masks, transforms)
    brown_low = -brown_translation[:2, 2]
    brown_high = brown_low+np.asarray(brown_size)
    vertices = np.concatenate([grid[1].reshape(-1, 2) for grid in grids])
    low = np.floor(np.minimum(vertices.min(axis=0), brown_low)-2).astype(int)
    high = np.ceil(np.maximum(vertices.max(axis=0)+3, brown_high)).astype(int)
    size = (high-low).astype(int)
    if np.any(size <= 0) or not np.isfinite(size).all():
        raise ValueError(f"APAP canvas is too large: {size.tolist()}")
    canvas_scale = min(1.0, float(np.sqrt(5_800_000/np.prod(size))))
    if canvas_scale < 1.0:
        size = np.maximum(1, np.floor(size*canvas_scale).astype(int))
    to_canvas = np.array([[canvas_scale, 0, -canvas_scale*low[0]],
                          [0, canvas_scale, -canvas_scale*low[1]],
                          [0, 0, 1]], np.float64)
    shape_hw = (int(size[1]), int(size[0]))
    warped, valid = [], []
    for image, mask, (src_grid, dst_grid) in zip(images, masks, grids):
        layer, layer_valid = _hdy_apap_render(
            image, mask, src_grid, dst_grid, shape_hw, to_canvas)
        warped.append(layer)
        valid.append(layer_valid)
    identity = np.eye(3, dtype=np.float64)
    panorama = _q3_multiband_blend(
        warped, valid, [identity]*8, identity,
        (shape_hw[1], shape_hw[0]), max_levels=6 if blend else 0)
    if return_details:
        return panorama, {
            "warped_images": warped, "warped_valid_masks": valid,
            "source_meshes": [item[0] for item in grids],
            "common_meshes": [item[1] for item in grids],
            "projected_images": images, "projected_valid_masks": masks,
            "global_transforms": transforms, "features": features,
            "ransac_edges": edges, "canvas_origin_xy": low,
            "canvas_scale": canvas_scale,
            "projection": projection, "blend": blend,
        }
    return panorama
