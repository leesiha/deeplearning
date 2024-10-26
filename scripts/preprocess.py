import os
from PIL import Image
import numpy as np
import cv2
import json

def calculate_vertical_diameter(points):
    """
    주어진 좌표에서 수직 직경을 계산하는 함수.
    points: [(x1, y1), (x2, y2), ...] 형태의 좌표 리스트
    """
    y_coords = [y for x, y in points]
    vertical_diameter = max(y_coords) - min(y_coords)
    return vertical_diameter

def calculate_cdr(json_path):
    """
    JSON 파일에서 OD와 OC의 좌표를 읽어 vCDR(Cup-to-Disc Ratio)을 계산.
    """
    with open(json_path, 'r') as file:
        annotation = json.load(file)

    od_diameter, oc_diameter = 0, 0

    for shape in annotation['shapes']:
        points = shape['points']

        if shape['label'] == 'disc':
            # 시신경 유두(OD) 수직 직경 계산
            od_diameter = calculate_vertical_diameter(points)
        elif shape['label'] == 'cup':
            # 시신경 함몰부(OC) 수직 직경 계산
            oc_diameter = calculate_vertical_diameter(points)

    # Cup-to-Disc 비율 계산
    if od_diameter > 0:
        vCDR = oc_diameter / od_diameter
    else:
        vCDR = 0

    return vCDR


def preprocess_vcdr(vcdr_value, min_val=0, max_val=1):
    """
    vCDR 수치를 정규화하는 함수
    - 0~1 범위로 변환
    
    Args:
        vcdr_value (float): vCDR 수치
        min_val (float): vCDR 최소값 (기본 0)
        max_val (float): vCDR 최대값 (기본 1)
        
    Returns:
        float: 정규화된 vCDR 값
    """
    # vCDR 값이 이미 min_val과 max_val 범위 내에 있으면 그대로 반환
    if min_val <= vcdr_value <= max_val:
        return vcdr_value

    # vCDR 값 정규화 (min_val~max_val 범위로)
    normalized_vcdr = (vcdr_value - min_val) / (max_val - min_val)
    return normalized_vcdr

                
# 이미지 전처리 함수 (정규화 및 크기 조정)
def preprocess_image(image_path, target_size=None, normalize=True):
    """
    이미지를 불러와서 정규화하는 함수 (크기 조정 필요 시 적용)
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple, optional): (width, height)로 이미지 크기 조정 (None일 경우 크기 조정 안함)
        normalize (bool): 0~255 값을 0~1로 정규화할지 여부
    
    Returns:
        numpy array: 전처리된 이미지
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # 크기 조정이 필요한 경우
    if target_size:
        image = cv2.resize(image, target_size)

    # 정규화 (0~255 값을 0~1로 변환)
    if normalize:
        image = image / 255.0

    return image

# 특정 데이터 세트(train/validate)에 대한 전처리


def preprocess_data(data_dir='data/train', resize_value=(224, 224)):
    """
    특정 데이터 디렉토리 내 모든 이미지를 전처리하는 함수
    - square, cropped, nerve_removed 디렉토리 내 이미지를 처리
    
    Args:
        data_dir (str): 데이터 디렉토리 경로 (train 또는 validate)
        target_size_cropped (tuple): cropped 및 nerve_removed 이미지 크기 조정 (width, height)
    
    Returns:
        dict: 전처리된 이미지 배열 딕셔너리
    """
    categories = {
        'square': resize_value,
        'cropped': resize_value,
        'nerve_removed': resize_value
    }
    all_preprocessed_data = {}
    
    # square, cropped, nerve_removed 디렉토리 내 이미지 전처리
    for category, target_size in categories.items():
        category_dir = os.path.join(data_dir, category)
        # print(f"Preprocessing {category_dir}...")
        preprocessed_images = {}

        for subdir, _, files in os.walk(category_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = os.path.join(subdir, file)
                    preprocessed_image = preprocess_image(
                        image_path, target_size)
                    preprocessed_images[file] = preprocessed_image

        all_preprocessed_data[category] = preprocessed_images
    
    # vCDR 처리
    vCDR_dir = os.path.join(data_dir, 'vCDR')
    # print(f"Preprocessing {vCDR_dir}...")
    vCDR_values = {}
    for subdir, _, files in os.walk(vCDR_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                vCDR_value = calculate_cdr(json_path)
                vCDR_values[file] = vCDR_value

    all_preprocessed_data['vCDR'] = vCDR_values

    return all_preprocessed_data
