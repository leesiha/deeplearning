import argparse
import os
from scripts.train import train_model
from scripts.generate import generate_text
from scripts.hyperparameter_tuning import tune_hyperparameters


def main():
    """
    명령어 기반 실행 구조
    이 함수는 명령줄에서 입력된 명령어(train, generate, tune)에 따라
    모델 학습, 텍스트 생성, 또는 하이퍼파라미터 튜닝을 실행합니다.
    """
    # 1. ArgumentParser 설정
    parser = argparse.ArgumentParser(
        description="Language Model CLI for Training, Generating, and Tuning"
    )
    # 2. Subparsers를 통해 명령어 분기 설정
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 2-1. train 명령어: 모델 학습
    train_parser = subparsers.add_parser("train", help="Train the language model")
    train_parser.add_argument("--train_dir", type=str, default="./data/Training/01.원천데이터/TS_1.발화단위평가_기술_과학", help="Path to the training data directory")
    train_parser.add_argument("--val_dir", type=str, default="./data/Validation/01.원천데이터/VS_1.발화단위평가_기술_과학", help="Path to the validation data directory")
    train_parser.add_argument("--model", type=str, choices=['attention', 'seq2seq', 'peeky'], default="attention", help="Model type: 'attention', 'seq2seq', or 'peeky'")
    train_parser.add_argument("--wordvec_size", type=int, default=16, help="Word vector size")
    train_parser.add_argument("--hidden_size", type=int, default=256, help="Hidden state size")
    train_parser.add_argument("--max_epoch", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--max_grad", type=float, default=5.0, help="Max gradient clipping value")
    train_parser.add_argument("--save_path", type=str, default="saved_models/model_checkpoint.pkl", help="Path to save the trained model")  # 학습된 모델 저장 경로
    train_parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    # 2-2. generate 명령어: 텍스트 생성
    generate_parser = subparsers.add_parser("generate", help="Generate text with the trained model")
    generate_parser.add_argument("--model_path", type=str, default="saved_models/model_checkpoint.pkl", help="Path to the trained model")  # 학습된 모델 경로
    generate_parser.add_argument("--start_text", type=str, default="Hello", help="Starting text for generation")  # 텍스트 생성 시작 문구
    generate_parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the generated text")  # 생성 텍스트 최대 길이

    # 2-3. tune 명령어: 하이퍼파라미터 튜닝
    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters for the model")
    tune_parser.add_argument("--iterations", type=int, default=20, help="Number of tuning iterations")  # 튜닝 반복 횟수
    tune_parser.add_argument("--save_best_model", type=bool, default=True, help="Save the best model after tuning")  # 최적의 모델 저장 여부

    # 3. 명령 파싱
    args = parser.parse_args()

    # 4. 명령어 처리
    if args.command == "train":
        # train 명령어 실행: 모델 학습 수행
        print("모델 학습을 시작합니다.")
        train_model(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            model_type=args.model,
            wordvec_size=args.wordvec_size,
            hidden_size=args.hidden_size,
            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            max_grad=args.max_grad,
            save_path=args.save_path,
            learning_rate=args.learning_rate
        )
        print("모델 학습 완료.")
    elif args.command == "generate":
        # generate 명령어 실행: 텍스트 생성 수행
        if not os.path.exists(args.model_path):  # 모델 파일 존재 여부 확인
            print(f"모델이 존재하지 않습니다. '{args.model_path}' 경로에 모델이 없습니다.")
            print("모델 학습을 시작합니다.")  # 모델이 없으면 학습 자동 수행
            train_model(epochs=10, batch_size=32, learning_rate=0.01, save_path=args.model_path)
            print("모델 학습 완료.")
        print("텍스트 생성을 시작합니다.")
        generate_text(model_path=args.model_path, start_text=args.start_text, max_length=args.max_length)
        print("텍스트 생성 완료.")
    elif args.command == "tune":
        # tune 명령어 실행: 하이퍼파라미터 튜닝 수행
        print("하이퍼파라미터 튜닝을 시작합니다.")
        tune_hyperparameters(iterations=args.iterations, save_best_model=args.save_best_model)
        print("하이퍼파라미터 튜닝 완료.")
    else:
        # 명령어 입력이 없거나 잘못된 경우 도움말 출력
        parser.print_help()

if __name__ == "__main__":
    main()
