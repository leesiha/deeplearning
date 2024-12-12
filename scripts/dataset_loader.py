import os
import json
import csv
import shutil
from typing import Dict, List


def load_labels(csv_path: str) -> Dict[str, int]:
    """
    CSV 파일에서 레이블을 불러오는 함수 (pandas 없이 csv 모듈 사용)
    
    Args:
        csv_path (str): 레이블이 포함된 CSV 파일 경로
    
    Returns:
        dict: 파일 이름과 레이블을 매핑한 딕셔너리
    """
    labels_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 첫 번째 줄(헤더) 스킵
        for row in reader:
            filename, label = row
            labels_dict[filename] = int(label)  # 레이블을 정수형으로 변환
    return labels_dict


def load_json(json_path: str) -> Dict:
    """
    JSON 파일을 읽어 딕셔너리로 반환.

    Args:
        json_path (str): JSON 파일 경로
    
    Returns:
        dict: JSON 파일 내용
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def copy_files(
    file_list: List[str],
    input_dir: str,
    output_dir: str,
    dir_type: str,
    cropped_dir: str = None,
    square_dir: str = None,
    nerve_removed_dir: str = None,
    copy_all: bool = True
):
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
    target_dir = os.path.join(output_dir, dir_type)
    os.makedirs(target_dir, exist_ok=True)

    for file_name in file_list:
        # 원본 디렉토리에서 파일 복사
        original_path = os.path.join(input_dir, file_name)
        if os.path.exists(original_path):
            shutil.copy(original_path, target_dir)

        if copy_all:
            # 추가 디렉토리에서 파일 복사
            if cropped_dir:
                cropped_path = os.path.join(cropped_dir, file_name)
                if os.path.exists(cropped_path):
                    shutil.copy(cropped_path, target_dir)

            if square_dir:
                square_path = os.path.join(square_dir, file_name)
                if os.path.exists(square_path):
                    shutil.copy(square_path, target_dir)

            if nerve_removed_dir:
                nerve_removed_path = os.path.join(nerve_removed_dir, file_name)
                if os.path.exists(nerve_removed_path):
                    shutil.copy(nerve_removed_path, target_dir)


def prepare_datasets(input_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    데이터셋을 train/validate/test로 분할하여 복사
    
    Args:
        input_dir (str): 원본 데이터 디렉토리 경로
        output_dir (str): 분할된 데이터 저장 디렉토리 경로
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
    """
    all_files = os.listdir(input_dir)
    total_files = len(all_files)

    random.shuffle(all_files)

    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    copy_files(train_files, input_dir, output_dir, dir_type='train')
    copy_files(val_files, input_dir, output_dir, dir_type='validate')
    copy_files(test_files, input_dir, output_dir, dir_type='test')
