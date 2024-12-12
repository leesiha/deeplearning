# coding: utf-8
import argparse
import os
import glob
import json
from models.peeky_seq2seq import PeekySeq2seq
from models.seq2seq import Seq2seq
from models.attention_seq2seq import AttentionSeq2seq
from common.util import eval_seq2seq
from common.trainer import Trainer
from common.optimizer import Adam
import matplotlib.pyplot as plt
import numpy as np
import csv


def load_json_data_from_directory(data_dir):
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
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):  # JSON이 리스트일 경우
                all_sentences.extend(data)
            elif isinstance(data, dict):  # JSON이 딕셔너리일 경우
                all_sentences.extend(data.values())
            else:
                raise ValueError(f"Unexpected JSON structure in file {file}")

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
    
    Args:
        sequences (list of list): 시퀀스 데이터
        maxlen (int): 패딩 후 시퀀스의 최대 길이
        padding (str): 'post' 또는 'pre'
        value (int): 패딩에 사용할 값

    Returns:
        np.array: 패딩된 시퀀스 배열
    """
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if padding == 'post':
                padded.append(seq[:maxlen])
            else:
                padded.append(seq[-maxlen:])
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                padded.append(seq + [value] * pad_len)
            else:
                padded.append([value] * pad_len + seq)
    return np.array(padded)

def load_tsv_data_from_directory(data_dir):
    """
    주어진 디렉토리에서 TSV 파일을 읽어 데이터를 로드하는 함수
    Args:
        data_dir (str): 데이터가 저장된 디렉토리 경로
    Returns:
        tuple: (x_data, t_data), char_to_id, id_to_char
    """
    # TSV 파일 경로 탐색
    tsv_files = glob.glob(os.path.join(data_dir, "**/*.tsv"), recursive=True)

    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in directory {data_dir}")

    # 데이터 읽기
    all_sentences = []
    for file in tsv_files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                input_sentence, target_sentence = row[0], row[1]
                all_sentences.append((input_sentence, target_sentence))

    # 문자 집합 생성
    char_set = sorted(set("".join([inp + tgt for inp, tgt in all_sentences])))
    char_to_id = {char: i for i, char in enumerate(char_set)}
    id_to_char = {i: char for char, i in char_to_id.items()}

    # 데이터를 ID 형태로 변환
    x_data = [[char_to_id[char] for char in inp] for inp, _ in all_sentences]
    t_data = [[char_to_id[char] for char in tgt] for _, tgt in all_sentences]

    # 최대 길이 계산 및 패딩
    max_len = max(max(len(x) for x in x_data), max(len(t) for t in t_data))
    x_data = pad_sequences(x_data, maxlen=max_len, padding='post', value=0)
    t_data = pad_sequences(t_data, maxlen=max_len, padding='post', value=0)

    return np.array(x_data), np.array(t_data), char_to_id, id_to_char


def train_model(train_dir, val_dir, model_type, wordvec_size, hidden_size, max_epoch, batch_size, max_grad, save_path, learning_rate):
    """
    모델 학습 함수
    Args:
        train_dir (str): 학습 데이터 디렉토리 경로
        val_dir (str): 검증 데이터 디렉토리 경로
        model_type (str): 사용할 모델의 유형 ('attention', 'seq2seq', 'peeky')
        wordvec_size (int): 단어 벡터 크기
        hidden_size (int): LSTM 등 숨겨진 상태 크기
        max_epoch (int): 학습 에폭 수
        batch_size (int): 배치 크기
        max_grad (float): 기울기 클리핑 최대값
        save_path (str): 학습된 모델 저장 경로
        learning_rate (float): 학습률
    """
    # 학습 데이터 로드
    print("Loading training data...")
    x_train, t_train, char_to_id, id_to_char = load_tsv_data_from_directory(
        train_dir)

    # 검증 데이터 로드
    print("Loading validation data...")
    x_val, t_val, _, _ = load_tsv_data_from_directory(val_dir)

    # 입력 문장 반전
    x_train, x_val = x_train[:, ::-1], x_val[:, ::-1]

    # 단어 사전 크기 계산
    vocab_size = len(char_to_id)

    # 모델 선택
    if model_type == 'attention':
        model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
    elif model_type == 'seq2seq':
        model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    elif model_type == 'peeky':
        model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
    else:
        raise ValueError(
            f"Invalid model type: {model_type}. Choose from 'attention', 'seq2seq', or 'peeky'.")

    # 옵티마이저에 학습률 설정
    optimizer = Adam(lr=learning_rate)

    # Trainer 객체 생성 (추가된 인자 포함)
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        print(f"Epoch {epoch + 1}/{max_epoch}")
        trainer.fit(x_train, t_train, max_epoch=max_epoch,
                    batch_size=batch_size, max_grad=max_grad)


        # 검증 데이터 평가
        print(f"Start validation for epoch {epoch + 1}/{max_epoch}")
        correct_num = 0

        for i in range(len(x_val)):
            question, correct = x_val[[i]], t_val[[i]]
            verbose = i < 10
            print(f"Validating sample {i+1}/{len(x_val)}...")

            # Eval 호출
            try:
                result = eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse=True)
                correct_num += result
            except Exception as e:
                print(f"Error occurred during validation at sample {i+1}: {e}")
                break


        acc = float(correct_num) / len(x_val)
        acc_list.append(acc)
        print(f"정확도: {acc * 100:.3f}%")

    # 모델 저장
    model.save_params(save_path)
    print(f"모델이 저장되었습니다: {save_path}")

    # 정확도 그래프 출력
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, marker='o')
    plt.xlabel('에폭')
    plt.ylabel('정확도')
    plt.ylim(-0.05, 1.05)
    plt.show()