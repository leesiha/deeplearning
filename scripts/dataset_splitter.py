import os
import random
import shutil

# 파일 리스트 생성 함수


def get_image_files(input_dir):
    """
    해당 디렉토리에서 이미지 파일 목록을 가져오는 함수
    
    Args:
        input_dir (str): 이미지 파일이 저장된 디렉토리 경로
        
    Returns:
        list: 이미지 파일 목록
    """
    return [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# 이미지 데이터를 분할하는 함수


def split_files(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    파일 리스트를 train/val/test로 분할하는 함수
    
    Args:
        files (list): 이미지 파일 리스트
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
    
    Returns:
        tuple: (train_files, val_files, test_files)
    """
    random.shuffle(files)
    train_split = int(train_ratio * len(files))
    val_split = int(val_ratio * len(files)) + train_split

    train_files = files[:train_split]
    val_files = files[train_split:val_split]
    test_files = files[val_split:]

    return train_files, val_files, test_files

# 출력 디렉토리 생성 함수


def create_output_dirs(output_dir, dirs, subdirs):
    """
    출력 디렉토리를 생성하는 함수. 디렉토리가 이미 존재하는 경우 삭제 후 생성
    
    Args:
        output_dir (str): 출력 디렉토리 경로
        dirs (list): train/validate/test 디렉토리 리스트
        subdirs (list): original/cropped/square/nerve_removed 디렉토리 리스트
    """

    for dir in dirs:
        for subdir in subdirs:
            if os.path.exists(os.path.join(output_dir, dir, subdir)):
                shutil.rmtree(os.path.join(output_dir, dir, subdir))
            os.makedirs(os.path.join(output_dir, dir, subdir), exist_ok=True)

# 파일 복사 함수


def copy_files(file_list, input_dir, output_dir, dir_type, cropped_dir=None, square_dir=None, nerve_removed_dir=None, copy_all=True):
    """
    파일을 원본에서 목적지로 복사하는 함수
    
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
        if os.path.exists(os.path.join(input_dir, img)):
            shutil.copy(os.path.join(input_dir, img), os.path.join(
                output_dir, dir_type, 'original', img))
        else:
            print(f"Original image not found for {img}")

        if copy_all:
            # vCDR 파일 복사
            json_file = img_base_name + '.json'
            if os.path.exists(os.path.join(input_dir, json_file)):
                shutil.copy(os.path.join(input_dir, json_file), os.path.join(
                    output_dir, dir_type, 'vCDR', json_file))
            else:
                print(f"JSON file not found for {img_base_name}")

            # 크롭된 이미지 복사
            cropped_img = img_base_name + '.jpg'
            if cropped_dir and os.path.exists(os.path.join(cropped_dir, cropped_img)):
                shutil.copy(os.path.join(cropped_dir, cropped_img), os.path.join(
                    output_dir, dir_type, 'cropped', cropped_img))
            else:
                print(f"Cropped image not found for {img_base_name}")

            # 정사각형 이미지 복사
            square_img = img_base_name + '.jpg'
            if square_dir and os.path.exists(os.path.join(square_dir, square_img)):
                shutil.copy(os.path.join(square_dir, square_img), os.path.join(
                    output_dir, dir_type, 'square', square_img))
            else:
                print(f"Square image not found for {img_base_name}")

            # 신경 제거된 이미지 복사
            nerve_removed_img = img_base_name + '.jpg'
            if nerve_removed_dir and os.path.exists(os.path.join(nerve_removed_dir, nerve_removed_img)):
                shutil.copy(os.path.join(nerve_removed_dir, nerve_removed_img), os.path.join(
                    output_dir, dir_type, 'nerve_removed', nerve_removed_img))
            else:
                print(f"Nerve removed image not found for {img_base_name}")

# 메인 데이터 분할 및 복사 작업 함수


def split_and_copy_data(input_dir='G1020/Images', output_dir='data/', cropped_dir='G1020/Images_Cropped/img', square_dir='G1020/Images_Square', nerve_removed_dir='G1020/NerveRemoved_Images', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    전체 데이터 분할 및 파일 복사 작업을 수행하는 메인 함수
    
    Args:
        input_dir (str): 입력 이미지 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
        cropped_dir (str): 크롭된 이미지가 저장된 디렉토리 경로
        square_dir (str): 정사각형 이미지가 저장된 디렉토리 경로
        nerve_removed_dir (str): 신경 제거된 이미지가 저장된 디렉토리 경로
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
    """
    # 파일 목록 생성
    images = get_image_files(input_dir)

    # 데이터 분할
    train_files, val_files, test_files = split_files(
        images, train_ratio, val_ratio, test_ratio)

    # 디렉토리 생성
    dirs = ['train', 'validate', 'test']
    subdirs = ['original', 'vCDR', 'cropped', 'square', 'nerve_removed']
    # train/validate에 모든 subdir 생성
    create_output_dirs(output_dir, ['train', 'validate'], subdirs)
    create_output_dirs(output_dir, ['test'], ['original'])  # test에 original만 생성

    # 파일 복사 (train, validate는 모든 subdirs, test는 original만 복사)
    copy_files(train_files, input_dir, output_dir, 'train',
               cropped_dir, square_dir, nerve_removed_dir, copy_all=True)
    copy_files(val_files, input_dir, output_dir, 'validate',
               cropped_dir, square_dir, nerve_removed_dir, copy_all=True)
    copy_files(test_files, input_dir, output_dir, 'test',
               copy_all=False)  # test에는 original만 복사
