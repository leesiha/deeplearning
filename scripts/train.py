# coding: utf-8
import argparse
import os
import glob
import json
import logging
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from models.peeky_seq2seq import PeekySeq2seq
from models.seq2seq import Seq2seq
from models.attention_seq2seq import AttentionSeq2seq
from common.util import eval_seq2seq
from common.trainer import Trainer
from common.optimizer import Adam

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic, Linux: Nanum Gothic)
rcParams['font.family'] = 'AppleGothic'  # Mac
# rcParams['font.family'] = 'Malgun Gothic'  # Windows
# rcParams['font.family'] = 'Nanum Gothic'  # Linux

# 유니코드 - 기호 깨짐 방지
rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('train.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_hyperparameters(
    wordvec_size: int,
    hidden_size: int,
    learning_rate: float,
    batch_size: int,
    max_epoch: int
) -> None:
    """하이퍼파라미터 유효성 검사"""
    if wordvec_size <= 0:
        raise ValueError("단어 벡터 크기는 양수여야 합니다.")
    if hidden_size <= 0:
        raise ValueError("은닉층 크기는 양수여야 합니다.")
    if not (0 < learning_rate < 1):
        raise ValueError("학습률은 0과 1 사이여야 합니다.")
    if batch_size < 1:
        raise ValueError("배치 크기는 1 이상이어야 합니다.")
    if max_epoch < 1:
        raise ValueError("최대 에폭 수는 1 이상이어야 합니다.")


def load_json_data_from_directory(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    주어진 디렉토리에서 JSON 파일을 읽어 데이터를 로드하는 함수
    Args:
        data_dir (str): 데이터가 저장된 디렉토리 경로
    Returns:
        tuple: (x_data, t_data), char_to_id, id_to_char
    """
    # JSON 파일 경로 탐색
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {data_dir}")

    # 데이터 읽기
    all_sentences = []
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_sentences.extend(data)
                elif isinstance(data, dict):
                    all_sentences.extend(data.values())
                else:
                    logger.warning(f"Unexpected JSON structure in file {file}")
        except json.JSONDecodeError:
            logger.error(f"JSON 디코딩 오류: {file}")

    # 문자 집합 생성
    char_to_id = {char: i for i, char in enumerate(
        sorted(set("".join(all_sentences))))}
    id_to_char = {i: char for char, i in char_to_id.items()}

    # 데이터를 ID 형태로 변환
    x_data = [[char_to_id[char] for char in sentence]
              for sentence in all_sentences]
    t_data = x_data  # 예제: 입력과 출력이 동일한 구조로 가정

    return np.array(x_data), np.array(t_data), char_to_id, id_to_char


def pad_sequences(sequences, maxlen, padding='post', value=0):
    """
    시퀀스 데이터를 패딩하여 동일한 길이로 맞춥니다.
    """
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded.append(seq[:maxlen] if padding == 'post' else seq[-maxlen:])
        else:
            pad_len = maxlen - len(seq)
            padded.append(seq + [value] * pad_len if padding ==
                          'post' else [value] * pad_len + seq)
    return np.array(padded)


def load_tsv_data_from_directory(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    주어진 디렉토리에서 TSV 파일을 읽어 데이터를 로드하는 함수
    """
    tsv_files = glob.glob(os.path.join(data_dir, "**/*.tsv"), recursive=True)

    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in directory {data_dir}")

    all_sentences = []
    for file in tsv_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    input_sentence, target_sentence = row[0], row[1]
                    all_sentences.append((input_sentence, target_sentence))
        except Exception as e:
            logger.error(f"TSV 파일 읽기 중 오류 발생: {file}, {e}")

    char_set = sorted(set("".join([inp + tgt for inp, tgt in all_sentences])))
    char_to_id = {char: i for i, char in enumerate(char_set)}
    id_to_char = {i: char for char, i in char_to_id.items()}

    x_data = [[char_to_id[char] for char in inp] for inp, _ in all_sentences]
    t_data = [[char_to_id[char] for char in tgt] for _, tgt in all_sentences]

    max_len = max(max(len(x) for x in x_data), max(len(t) for t in t_data))
    x_data = pad_sequences(x_data, maxlen=max_len, padding='post', value=0)
    t_data = pad_sequences(t_data, maxlen=max_len, padding='post', value=0)

    return np.array(x_data), np.array(t_data), char_to_id, id_to_char


def train_model(
    train_dir: str = "./data/Training/01.원천데이터/TS_1.발화단위평가_기술_과학",
    val_dir: str = "./data/Validation/01.원천데이터/VS_1.발화단위평가_기술_과학",
    model_type: str = 'attention',
    wordvec_size: int = 16,
    hidden_size: int = 256,
    max_epoch: int = 10,
    batch_size: int = 32,
    max_grad: float = 5.0,
    save_path: str = "saved_models/model_checkpoint.pkl",
    learning_rate: float = 0.01
):
    """
    모델 학습 함수
    """
    # 결과 저장 경로 설정
    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)
    hyperparams_path = os.path.join(base_dir, "hyperparams.json")
    vocab_save_path = os.path.join(base_dir, "vocab.json")
    results_path = f'{save_path}_results.json'
    training_plot_path = f'{save_path}_training_plot.png'

    # 하이퍼파라미터 유효성 검사
    validate_hyperparameters(wordvec_size, hidden_size,
                             learning_rate, batch_size, max_epoch)

    # 데이터 로드
    logger.info("학습 데이터 로드 중...")
    try:
        x_train, t_train, char_to_id, id_to_char = load_tsv_data_from_directory(
            train_dir)
    except Exception as e:
        logger.error(f"학습 데이터 로드 실패: {e}")
        raise

    logger.info("검증 데이터 로드 중...")
    try:
        x_val, t_val, _, _ = load_tsv_data_from_directory(val_dir)
    except Exception as e:
        logger.error(f"검증 데이터 로드 실패: {e}")
        raise

    # 입력 문장 반전
    x_train, x_val = x_train[:, ::-1], x_val[:, ::-1]
    vocab_size = len(char_to_id)

    # Vocabulary 저장
    vocab_data = {'word_to_id': char_to_id, 'id_to_word': id_to_char}
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=4, ensure_ascii=False)
    logger.info(f"Vocabulary 저장 완료: {vocab_save_path}")

    # 하이퍼파라미터 저장
    hyperparams = {
        'model_type': model_type,
        'wordvec_size': wordvec_size,
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epoch': max_epoch,
        'vocab_size': vocab_size
    }
    with open(hyperparams_path, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    logger.info(f"하이퍼파라미터 저장 완료: {hyperparams_path}")

    # 모델 생성
    model_factory = {
        'attention': AttentionSeq2seq,
        'seq2seq': Seq2seq,
        'peeky': PeekySeq2seq
    }
    model_class = model_factory.get(model_type)
    if not model_class:
        logger.error(f"지원되지 않는 모델 타입: {model_type}")
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}")

    model = model_class(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam(lr=learning_rate)
    trainer = Trainer(model, optimizer)

    # 학습
    acc_list: List[float] = []
    loss_list: List[float] = []

    logger.info("모델 학습 시작...")
    for epoch in tqdm(range(max_epoch), desc="학습 진행"):
        # 학습
        train_loss = trainer.fit(
            x_train, t_train,
            max_epoch=max_epoch,
            batch_size=batch_size,
            max_grad=max_grad
        )
        loss_list.append(train_loss)

        # 검증
        correct_num = 0
        for i in tqdm(range(len(x_val)), desc=f"에폭 {epoch+1} 검증", leave=False):
            question, correct = x_val[[i]], t_val[[i]]
            try:
                result = eval_seq2seq(
                    model, question, correct, id_to_char, verbose=False, is_reverse=True
                )
                correct_num += result
            except Exception as e:
                logger.error(f"검증 중 오류 발생: {e}")
                break

        # 정확도 계산
        acc = float(correct_num) / len(x_val)
        acc_list.append(acc)
        logger.info(
            f"에폭 {epoch+1}/{max_epoch}: 정확도 = {acc * 100:.3f}%, 손실 = {train_loss:.4f}")

    # 모델 저장
    model.save_params(save_path)
    logger.info(f"모델이 저장되었습니다: {save_path}")

    # 결과 저장
    results = {
        'accuracy_history': acc_list,
        'loss_history': loss_list
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"결과 저장 완료: {results_path}")

    # 학습 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(acc_list) + 1), acc_list, marker='o')
    plt.title('에폭별 정확도')
    plt.xlabel('에폭')
    plt.ylabel('정확도')
    plt.ylim(-0.05, 1.05)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', color='red')
    plt.title('에폭별 손실')
    plt.xlabel('에폭')
    plt.ylabel('손실')
    plt.tight_layout()
    plt.savefig(training_plot_path)
    plt.close()
    logger.info(f"학습 그래프 저장 완료: {training_plot_path}")

    logger.info("모델 학습 완료.")
    return results
