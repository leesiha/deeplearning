import os
import numpy as np
import json
from PIL import Image
import cv2

# 이미지 불러오기 함수


def load_image(image_path):
    """
    이미지 경로를 받아 이미지를 불러온 후, numpy 배열로 반환.
    """
    image = Image.open(image_path)
    return np.array(image)

# JSON 데이터를 기반으로 세그멘테이션 마스크 생성


def scale_points(points, original_size, new_size=(512, 512)):
    """
    좌표를 가로 세로 비율을 유지한 채 패딩된 512x512 이미지에 맞춰 스케일링.
    """
    width, height = original_size[1], original_size[0]  # 원본 이미지의 가로/세로 크기
    new_width, new_height = new_size[1], new_size[0]  # 새 이미지의 가로/세로 크기

    # 원본 비율 유지하며 좌표 스케일링
    scale = min(new_width / width, new_height / height)

    # 패딩 오프셋 계산
    x_offset = (new_width - width * scale) / 2
    y_offset = (new_height - height * scale) / 2

    scaled_points = [(int(x * scale + x_offset),
                      int(y * scale + y_offset)) for x, y in points]
    return scaled_points


def create_mask(json_path, original_size, new_size=(512, 512)):
    """
    original_size에 대응하는 정보를 new_size에 맞춰 좌표 변환한 후 마스크 생성.
    """
    mask = np.zeros(new_size, dtype=np.uint8)

    with open(json_path, 'r') as file:
        annotation = json.load(file)

    for shape in annotation['shapes']:
        points = np.array(shape['points'], dtype=np.int32)

        # 좌표 스케일링 적용
        scaled_points = scale_points(points, original_size, new_size)

        if shape['label'] == 'disc':
            cv2.fillPoly(
                mask, [np.array(scaled_points, dtype=np.int32)], color=1)  # OD 부분
        elif shape['label'] == 'cup':
            cv2.fillPoly(
                mask, [np.array(scaled_points, dtype=np.int32)], color=2)  # OC 부분

    return mask
