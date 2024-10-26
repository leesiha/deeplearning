import optuna
import numpy as np
from scripts.preprocess import preprocess_data
from scripts.dataset_splitter import split_and_copy_data
from scripts.dataset_loader import load_labels
from models.multiscale_convnet import MultiscaleConvNet
from common.trainer import Trainer
from models.multi_layer_net_extend import MultiLayerNetExtend


def objective(trial):
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

    # 하이퍼파라미터 범위 설정
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 64)

    # 모델 학습 및 검증 진행 (모델 정의 필요)
    model = MultiLayerNetExtend(
        input_size=input_size,
        hidden_size_list=hidden_size_list,
        output_size=output_size,
        activation='relu',
        weight_decay_lambda=weight_decay,
        use_dropout=True,
        dropout_ration=dropout_rate,
        use_batchnorm=True
    )
    accuracy = Trainer(
        network=model,
        x_train=train_images, t_train=train_labels,
        x_test=validate_images, t_test=validate_labels,
        epochs=20,
        mini_batch_size=batch_size,
        optimizer='Adam',
        optimizer_param={'lr': learning_rate},
        evaluate_sample_num_per_epoch=1000
    ).train()

    return accuracy


# Optuna 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"최적의 하이퍼파라미터: {study.best_params}")
