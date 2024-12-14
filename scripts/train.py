# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib import rcParams
from models.attention_seq2seq import AttentionSeq2seq
from models.seq2seq import Seq2seq
from models.peeky_seq2seq import PeekySeq2seq
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from common.np import *

import os
import logging
from typing import Tuple, Dict, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def load_tsv_data_from_directory(data_dir: str, sample_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Load sentences from a TSV directory, tokenize by spaces, and build a vocabulary at the word level.
    Optionally, randomly sample a fraction of the data.
    
    Args:
        data_dir (str): Directory containing TSV files.
        sample_fraction (float): Fraction of the data to sample (default=1.0, meaning no sampling).
    Returns:
        tuple: (x_data, t_data, word_to_id, id_to_word)
    """
    import glob
    import csv

    logger = logging.getLogger(__name__)

    # TSV 파일 검색
    tsv_files = glob.glob(os.path.join(data_dir, "**/*.tsv"), recursive=True)

    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in directory {data_dir}")

    all_sentences = []
    for file in tsv_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:  # Ensure sufficient columns
                        utterance_text = row[2]
                        all_sentences.append(utterance_text.strip())
        except Exception as e:
            logger.error(f"Error reading TSV file: {file}, {e}")

    # logger.info(f"Total sentences loaded: {len(all_sentences)}")

    # 데이터 샘플링
    if 0 < sample_fraction < 1.0:
        sample_size = int(len(all_sentences) * sample_fraction)
        sampled_indices = np.random.choice(len(all_sentences), sample_size, replace=False)
        all_sentences = [all_sentences[i] for i in sampled_indices.tolist()]
        logger.info(
            f"Sampled {sample_size} sentences from {len(all_sentences)}")

    # 단어 집합 생성
    all_words = [
        word for sentence in all_sentences for word in sentence.split()
    ]
    word_set = sorted(set(all_words))  # 고유 단어 집합
    word_to_id = {word: i for i, word in enumerate(word_set)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # logger.info(f"Vocabulary size: {len(word_to_id)}")
    # logger.info(f"Sample vocabulary: {list(word_to_id.items())[:10]}")

    # ID로 변환
    x_data = [[word_to_id[word] for word in sentence.split() if word in word_to_id]
              for sentence in all_sentences]

    # 패딩 처리
    max_len = max(len(seq) for seq in x_data) if x_data else 0
    x_data = pad_sequences(x_data, maxlen=max_len, padding='post', value=0)

    logger.info(f"Input data shape: {x_data.shape}")

    # 타겟 데이터는 입력과 동일
    t_data = x_data

    return np.array(x_data), t_data, word_to_id, id_to_word

def train_model(
    train_dir: str,
    val_dir: str,
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
    모델 학습 함수 (trainer.py 원본 코드 사용)
    """
    # 데이터 로드
    x_train, t_train, char_to_id, id_to_char = load_tsv_data_from_directory(train_dir)
    x_val, t_val, _, _ = load_tsv_data_from_directory(val_dir)

    if batch_size > len(x_train):
        batch_size = len(x_train)
        logger.warning(
            f"Batch size adjusted to {batch_size} as it exceeded dataset size.")

    # 모델 준비
    model_factory = {
        'attention': AttentionSeq2seq,
        'seq2seq': Seq2seq,
        'peeky': PeekySeq2seq
    }
    model_class = model_factory.get(model_type)
    if not model_class:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = model_class(len(char_to_id), wordvec_size, hidden_size)
    optimizer = Adam(lr=learning_rate)
    trainer = Trainer(model, optimizer)

    # 학습 수행
    try:
        trainer.fit(
            x_train, t_train,
            max_epoch=max_epoch,
            batch_size=batch_size,
            max_grad=max_grad
        )
    except ValueError as e:
        logger.error(f"Training failed: {e}")
        return

    # 검증 수행
    correct = 0
    total = len(x_val)
    for i in range(total):
        question, answer = x_val[[i]], t_val[[i]]
        try:
            result = eval_seq2seq(model, question, answer,
                                  id_to_char, is_reverse=True)
            correct += result
        except Exception as e:
            logger.error(f"Validation error at sample {i}: {e}")
            continue

    accuracy = correct / total
    logger.info(f"Validation Accuracy: {accuracy:.2%}")

    # 모델 저장
    model.save_params(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    # 테스트용 데이터 디렉토리 경로
    test_data_dir = "./data/Training/01.원천데이터/TS_1.발화단위평가_기술_과학"

    # 테스트 실행
    try:
        logger.info("TSV 데이터 로드 테스트 시작...")
        x_data, t_data, char_to_id, id_to_char = load_tsv_data_from_directory(test_data_dir)

        # 로드된 데이터 정보 출력
        logger.info(f"입력 데이터 샘플 크기: {x_data.shape}")
        logger.info(f"타겟 데이터 샘플 크기: {t_data.shape}")
        logger.info(f"단어 집합 크기: {len(char_to_id)}")
        logger.info(f"단어 집합 예시: {list(char_to_id.items())[:10]}")

        # 첫 번째 데이터 샘플 출력
        logger.info(f"첫 번째 입력 데이터 (ID): {x_data[0]}")
        logger.info(f"첫 번째 타겟 데이터 (ID): {t_data[0]}")
        logger.info(
            f"첫 번째 입력 데이터 (문자): {''.join([id_to_char[id] for id in x_data[0] if id in id_to_char])}")
        logger.info(
            f"첫 번째 타겟 데이터 (문자): {''.join([id_to_char[id] for id in t_data[0] if id in id_to_char])}")

        logger.info("TSV 데이터 로드 테스트 완료.")
    except Exception as e:
        logger.error(f"TSV 데이터 로드 테스트 실패: {e}")
