import numpy as np
from scripts.preprocess import preprocess_data
from scripts.dataset_splitter import split_and_copy_data
from scripts.dataset_loader import load_labels
from models.multiscale_convnet import MultiscaleConvNet
from common.trainer import Trainer
from models.multi_layer_net_extend import MultiLayerNetExtend


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
    input_size = 224 * 224 * 3  # 224x224 이미지 크기와 3개의 채널(RGB)
    hidden_size_list = [100, 100, 100]  # 은닉층 뉴런 수
    output_size = 2  # 출력층 뉴런 수 (이진 분류)

    model = MultiLayerNetExtend(
        input_size=input_size,
        hidden_size_list=hidden_size_list,
        output_size=output_size,
        activation='relu',  # 활성화 함수
        weight_decay_lambda=0.01,  # 가중치 감소 (L2 규제)
        use_dropout=True,  # 드롭아웃 사용
        dropout_ration=0.5,  # 드롭아웃 비율
        use_batchnorm=True  # 배치 정규화 사용
    )

    trainer = Trainer(
        network=model,
        x_train=train_images, t_train=train_labels,
        x_test=validate_images, t_test=validate_labels,
        epochs=20, mini_batch_size=32,
        optimizer='Adam', optimizer_param={'lr': 0.001},
        verbose=True
    )

    trainer.train()
    print("모델 학습 완료.")

    # 4. 테스트 데이터로 평가 (test set으로 예측 및 성능 확인)
    test_data = preprocess_data(data_dir='data/test')
    test_images = np.array(list(test_data['square'].values()))
    test_labels = np.array([labels_dict[file]
                            for file in test_data['square'].keys()])
    test_images = test_images.transpose(0, 3, 1, 2)
    print("테스트 데이터 준비 완료.")

    # 5. 테스트 데이터 평가 직접 구현
    test_predictions = trainer.network.predict(test_images)  # 테스트 데이터 예측
    test_predictions = np.argmax(test_predictions, axis=1)  # 가장 높은 확률의 클래스로 변환

    # 정확도 계산
    test_accuracy = np.mean(test_predictions == test_labels)
    print(f"최종 테스트 정확도: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
