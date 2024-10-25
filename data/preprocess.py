import os
from PIL import Image
import numpy as np
from data.dataset_loader import load_image, create_mask

# 마스크 저장 함수


def save_processed_image(mask, save_dir, image_name):
    """
    전처리된 이미지를 저장하는 함수.
    """
    mask_save_path = os.path.join(save_dir, image_name)
    mask_scaled = (mask * 127).astype(np.uint8)  # 1 -> 127, 2 -> 254
    Image.fromarray(mask_scaled).save(mask_save_path)

# 모든 이미지 전처리 함수


def process_all_images(input_dir, output_dir):
    """
    모든 이미지를 처리하여 지정된 출력 디렉토리에 저장.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            json_filename = filename.replace(".jpg", ".json")
            json_path = os.path.join(input_dir, json_filename)

            if os.path.exists(json_path):
                # 이미지와 마스크 생성 및 전처리
                image = load_image(image_path)
                original_size = image.shape[:2]
                mask = create_mask(json_path, original_size)

                # 저장 코드
                save_processed_image(mask, output_dir, filename)
                print(f"Processed {filename}")
