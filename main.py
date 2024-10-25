from data.preprocess import process_all_images

def main():
    # 1. 데이터 전처리
    input_dir = 'G1020/Images'
    output_dir = 'G1020/Images_Preprocessed'
    process_all_images(input_dir, output_dir)
    print("데이터 전처리 완료.")

    # 2. 모델 학습
    train_data_dir = 'G1020/Images_Preprocessed'
    # 구현 필요
    print("모델 학습 완료.")

if __name__ == "__main__":
    main()
