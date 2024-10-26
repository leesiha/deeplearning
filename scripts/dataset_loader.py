import os
import numpy as np
import json
from PIL import Image
import random
import shutil
import csv


def load_labels(csv_path):
    """
    CSV 파일에서 레이블을 불러오는 함수 (pandas 없이 csv 모듈 사용)
    
    Args:
        csv_path (str): 레이블이 포함된 CSV 파일 경로
    
    Returns:
        dict: 파일 이름과 레이블을 매핑한 딕셔너리
    """
    labels_dict = {}

    # CSV 파일을 열고 읽음
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 첫 번째 줄(헤더) 스킵
        for row in reader:
            filename, label = row
            labels_dict[filename] = int(label)  # 레이블을 정수형으로 변환

    return labels_dict


def load_image(image_path):
    """
    이미지 경로를 받아 이미지를 불러온 후, numpy 배열로 반환.
    """
    image = Image.open(image_path)
    return np.array(image)

def load_json(json_path):
    """
    JSON 파일을 읽어 딕셔너리로 반환.
    """
    with open(json_path, 'r') as file:
        annotation = json.load(file)
    return annotation

def split_data(input_dir='G1020/Images', output_dir='data/', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    images = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    train_split = int(train_ratio * len(images))
    val_split = int(val_ratio * len(images)) + train_split

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # Create output directories
    dirs = ['train', 'val', 'test']
    subdirs = ['original', 'vcdr', 'cropped', 'nerve_removed']
    for dir in dirs:
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir, dir), exist_ok=True)

    # Move files to corresponding directories
    for img_set, dirs in zip([train_images, val_images, test_images], ['train', 'val', 'test']):
        for img in img_set:
            json_file = img.replace('.jpg', '.json')
            # 이미지와 해당 json 파일을 복사
            if os.path.exists(os.path.join(input_dir, json_file)):
                shutil.copy(os.path.join(input_dir, img), os.path.join(
                    output_dir + subdirs[0], dirs, img))
                shutil.copy(os.path.join(input_dir, json_file), os.path.join(
                    output_dir + subdirs[1], dirs, json_file))
            else:
                print(f"JSON file not found for {img}")


def copy_image(img_base_name, img_dir, output_dir, subdir, dir_type):
    """
    이미지를 복사하는 함수 (경로에 따라 원본, 크롭, 정사각형, 신경 제거 이미지에 대해 사용 가능)
    
    Args:
        img_base_name (str): 이미지 파일명의 기본 이름 (확장자 제외)
        img_dir (str): 원본 이미지가 저장된 디렉토리 경로
        output_dir (str): 복사할 디렉토리 경로
        subdir (str): original, cropped, square, nerve_removed 중 하나
        dir_type (str): train/validate/test 중 하나
    """
    img = img_base_name + '.jpg'
    if os.path.exists(os.path.join(img_dir, img)):
        shutil.copy(os.path.join(img_dir, img), os.path.join(
            output_dir, dir_type, subdir, img))
    else:
        print(f"{subdir.capitalize()} image not found for {img_base_name}")

# 메인 복사 작업 함수


def copy_files(file_list, input_dir, output_dir, dir_type, cropped_dir=None, square_dir=None, nerve_removed_dir=None, copy_all=True):
    """
    파일을 원본에서 목적지로 복사하는 메인 함수
    
    Args:
        file_list (list): 복사할 파일 리스트
        input_dir (str): 원본 파일이 있는 디렉토리
        output_dir (str): 복사할 디렉토리 경로
        dir_type (str): train/validate/test 중 하나
        cropped_dir (str): 크롭된 이미지 디렉토리 경로
        square_dir (str): 정사각형 이미지 디렉토리 경로
        nerve_removed_dir (str): 신경 제거된 이미지 디렉토리 경로
        copy_all (bool): True일 경우 모든 subdirs를 복사, False일 경우 original만 복사
    """
    for img in file_list:
        img_base_name = os.path.splitext(img)[0]  # 파일명(확장자 제외)

        # Original 이미지 복사
        copy_image(img_base_name, input_dir, output_dir, 'original', dir_type)

        if copy_all:
            # 크롭된 이미지 복사
            copy_image(img_base_name, cropped_dir,
                       output_dir, 'cropped', dir_type)

            # 정사각형 이미지 복사
            copy_image(img_base_name, square_dir,
                       output_dir, 'square', dir_type)

            # 신경 제거된 이미지 복사
            copy_image(img_base_name, nerve_removed_dir,
                       output_dir, 'nerve_removed', dir_type)
