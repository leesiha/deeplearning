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

def load_json(json_path):
    """
    JSON 파일을 읽어 딕셔너리로 반환.
    """
    with open(json_path, 'r') as file:
        annotation = json.load(file)
    return annotation

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
    