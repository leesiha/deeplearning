import optuna
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

def objective(trial):
    # 0. 데이터 분할
    split_and_copy_data(input_dir='G1020/Images',
                        output_dir='data/',
                        cropped_dir='G1020/Images_Cropped/img',
                        square_dir='G1020/Images_Square',
                        nerve_removed_dir='G1020/NerveRemoved_Images')
    labels_dict = load_labels('G1020/G1020.csv')

    # 1. 데이터 전처리
    train_data = preprocess_data(data_dir='data/train')
    validate_data = preprocess_data(data_dir='data/validate')

    # 2. 데이터 준비
    train_square, train_cropped, train_nerve, train_vcdr, train_labels = \
        prepare_data_for_training(train_data, labels_dict)

    val_square, val_cropped, val_nerve, val_vcdr, val_labels = \
        prepare_data_for_training(validate_data, labels_dict)

    # 3. 모델 학습

    # 하이퍼파라미터 범위 설정
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 0, 0.1)

    # 모델 학습 및 검증 진행 (모델 정의 필요)
    model = MultimodalNet(
        image_size=224 * 224 * 3,  # 224x224 RGB 이미지
        vcdr_size=1,  # vCDR 값은 1차원
        hidden_size_list=[100, 100],  # 히든 레이어 크기
        output_size=2,
        activation='relu',
        weight_decay_lambda=weight_decay,
        use_dropout=True,
        dropout_ration=dropout_rate,
        use_batchnorm=True
    )

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    accuracy = Trainer(
        network=model,
        x_train=(train_square, train_cropped, train_nerve, train_vcdr),
        t_train=train_labels,
        x_test=(val_square, val_cropped, val_nerve, val_vcdr),
        t_test=val_labels,
        epochs=20,
        mini_batch_size=batch_size,
        optimizer='Adam',
        optimizer_param={'lr': learning_rate},
        verbose=False
    ).train()

    return accuracy


# Optuna 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"최적의 하이퍼파라미터: {study.best_params}")
