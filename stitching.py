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


def q1(img_paths):
    """Q1. planar(평면) 환경에서 2장 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    raise NotImplementedError


def q2(img_paths):
    """Q2. cylindrical(원통) 투영 후 3장 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    raise NotImplementedError


def q3(img_paths):
    """Q3. 직접 촬영한 8장을 cylindrical 투영 후 정합.

    입력 8장은 2×4 격자이며 img_paths에 raster order로 들어옵니다.
    (윗줄 왼쪽→오른쪽 1→4, 그다음 아랫줄 왼쪽→오른쪽 5→8)
        1  2  3  4
        5  6  7  8

    Returns: 정합 결과 이미지 (ndarray)
    """
    raise NotImplementedError


def e1(img_paths):
    """E1 (추가점수). Q3의 8장을 spherical(구면) 투영 후 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    raise NotImplementedError


def e2(img_paths):
    """E2 (추가점수). E1 결과에 blending을 적용해 자연스럽게 정합.

    Returns: 정합 결과 이미지 (ndarray)
    """
    raise NotImplementedError
