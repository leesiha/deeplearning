import numpy as np
from scripts.preprocess import preprocess_data
from scripts.dataset_splitter import split_and_copy_data
from scripts.dataset_loader import load_labels
from models.multimodal_net import MultimodalNet  # 새로운 모델 import
from common.trainer import Trainer


def prepare_data_for_training(data_dict, labels_dict):
    """
    전처리된 데이터를 학습에 적합한 형태로 변환
    """
    # 이미지 데이터 준비
    square_images = np.array(list(data_dict['square'].values()))
    cropped_images = np.array(list(data_dict['cropped'].values()))
    nerve_images = np.array(list(data_dict['nerve_removed'].values()))

    # vCDR 데이터 준비
    vcdr_values = np.array(list(data_dict['vCDR'].values()))
    vcdr_values = vcdr_values.reshape(-1, 1)  # (N, 1) 형태로 reshape

    # 레이블 준비
    labels = np.array([labels_dict[file]
                      for file in data_dict['square'].keys()])

    # 이미지 데이터 차원 변환 (N, H, W, C) -> (N, C, H, W)
    square_images = square_images.transpose(0, 3, 1, 2)
    cropped_images = cropped_images.transpose(0, 3, 1, 2)
    nerve_images = nerve_images.transpose(0, 3, 1, 2)

    return square_images, cropped_images, nerve_images, vcdr_values, labels


def main():
    # 0. 데이터 분할
    split_and_copy_data(input_dir='G1020/Images',
                        output_dir='data/',
                        cropped_dir='G1020/Images_Cropped/img',
                        square_dir='G1020/Images_Square',
                        nerve_removed_dir='G1020/NerveRemoved_Images')
    print("데이터 분할 완료.")

    labels_dict = load_labels('G1020/G1020.csv')
    print("레이블 데이터 로드 완료.")

    # 1. 데이터 전처리
    train_data = preprocess_data(data_dir='data/train')
    validate_data = preprocess_data(data_dir='data/validate')
    print("데이터 전처리 완료.")

    # 2. 데이터 준비
    train_square, train_cropped, train_nerve, train_vcdr, train_labels = \
        prepare_data_for_training(train_data, labels_dict)

    val_square, val_cropped, val_nerve, val_vcdr, val_labels = \
        prepare_data_for_training(validate_data, labels_dict)

    print("데이터 준비 완료.")

    # 3. 모델 학습
    model = MultimodalNet(
        image_size=224 * 224 * 3,  # 224x224 RGB 이미지
        vcdr_size=1,  # vCDR 값은 1차원
        hidden_size_list=[100, 100],  # 히든 레이어 크기
        output_size=2,  # 이진 분류
        activation='relu',
        use_dropout=True,
        use_batchnorm=True
    )

    trainer = Trainer(
        network=model,
        x_train=(train_square, train_cropped, train_nerve, train_vcdr),
        t_train=train_labels,
        x_test=(val_square, val_cropped, val_nerve, val_vcdr),
        t_test=val_labels,
        epochs=20,
        mini_batch_size=19,
        optimizer='Adam',
        optimizer_param={'lr': 0.01},
        verbose=True
    )

    trainer.train()
    print("모델 학습 완료.")

    # 4. 테스트 데이터로 평가
    test_data = preprocess_data(data_dir='data/test')
    test_square, test_cropped, test_nerve, test_vcdr, test_labels = \
        prepare_data_for_training(test_data, labels_dict)
    print("테스트 데이터 준비 완료.")

    # 5. 테스트 데이터 평가
    test_accuracy = model.accuracy(
        test_square, test_cropped, test_nerve, test_vcdr, test_labels
    )
    print(f"최종 테스트 정확도: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
