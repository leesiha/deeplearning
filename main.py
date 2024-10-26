import numpy as np
from scripts.preprocess import preprocess_data
from scripts.dataset_splitter import split_and_copy_data
from scripts.dataset_loader import load_labels
from models.multiscale_convnet import MultiscaleConvNet
from common.trainer import Trainer


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
    train_images = np.array(list(train_data['square'].values()))
    train_labels = np.array([labels_dict[file]
                            for file in train_data['square'].keys()])

    validate_images = np.array(list(validate_data['square'].values()))
    validate_labels = np.array([labels_dict[file]
                               for file in validate_data['square'].keys()])
    
    # 훈련 및 검증 데이터의 차원 변환 (N, H, W, C) -> (N, C, H, W)
    train_images = train_images.transpose(0, 3, 1, 2)
    validate_images = validate_images.transpose(0, 3, 1, 2)
    print("데이터 준비 완료.")

    # 3. 모델 학습
    input_shape = (3, 256, 256)  # 채널 수, 높이, 너비
    num_classes = 2  # 이진 분류

    # 모델 생성
    model = MultiscaleConvNet(input_dim=input_shape, num_classes=num_classes)

    # Trainer를 사용해 모델 학습
    trainer = Trainer(
        network=model,
        x_train=train_images, t_train=train_labels,
        x_test=validate_images, t_test=validate_labels,
        epochs=20, mini_batch_size=32,
        optimizer='adam', optimizer_param={'lr': 0.001},
        verbose=True
    )

    trainer.train()
    print("모델 학습 완료.")

    # 4. 테스트 데이터로 평가 (test set으로 예측 및 성능 확인)
    test_data = preprocess_data(data_dir='data/test')
    test_images, test_labels = np.array(
        test_data['square']), np.array(test_data['labels'])

    # 최종 테스트 평가
    test_loss, test_acc = trainer.network.evaluate(test_images, test_labels)
    print(f"최종 테스트 정확도: {test_acc:.4f}")


if __name__ == "__main__":
    main()
