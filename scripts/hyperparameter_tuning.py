# coding: utf-8
import itertools
import random
from common.trainer import Trainer
from models.language_model import Seq2seq # options: Rnnlm, BetterRnnlm, Seq2seq
from scripts.dataset_loader import load_json


def tune_hyperparameters(
    train_data,
    valid_data,
    param_grid=None,
    iterations=10,
    save_best_model=True,
    save_path="best_model.pkl"
):
    """
    하이퍼파라미터 튜닝 함수
    Args:
        train_data (tuple): 학습 데이터 (x_train, t_train)
        valid_data (tuple): 검증 데이터 (x_valid, t_valid)
        param_grid (dict): 튜닝할 하이퍼파라미터의 범위
        iterations (int): 랜덤 검색 방식에서의 반복 횟수
        save_best_model (bool): 최적 모델 저장 여부
        save_path (str): 최적 모델 저장 경로
    """
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.01, 0.001, 0.0001],
            "batch_size": [16, 32, 64],
            "hidden_size": [128, 256, 512],
            "dropout": [0.3, 0.5, 0.7]
        }

    # 파라미터 조합 생성
    param_combinations = list(itertools.product(*param_grid.values()))
    if len(param_combinations) > iterations:
        param_combinations = random.sample(param_combinations, iterations)

    best_accuracy = 0
    best_params = None
    best_model = None

    print("Starting hyperparameter tuning...")

    for params in param_combinations:
        # 현재 조합 파라미터 출력
        current_params = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {current_params}")

        # 모델 초기화
        model = LanguageModel(
            hidden_size=current_params["hidden_size"], dropout=current_params["dropout"])
        trainer = Trainer(
            model,
            optimizer="adam",
            optimizer_param={"lr": current_params["learning_rate"]},
            x_train=train_data[0],
            t_train=train_data[1],
            x_test=valid_data[0],
            t_test=valid_data[1],
            mini_batch_size=current_params["batch_size"],
            epochs=10,
            verbose=False
        )

        # 모델 학습
        trainer.train()

        # 검증 데이터로 성능 평가
        accuracy = model.accuracy(valid_data[0], valid_data[1])
        print(f"Validation accuracy: {accuracy:.4f}")

        # 최적 모델 갱신
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = current_params
            best_model = model

            # 최적 모델 저장
            if save_best_model:
                model.save_params(save_path)
                print(
                    f"New best model saved to {save_path} with accuracy: {accuracy:.4f}")

    print("Hyperparameter tuning complete.")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    return best_model, best_params, best_accuracy
