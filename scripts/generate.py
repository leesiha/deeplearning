# coding: utf-8
import sys
import os
import json
from models.rnnlm_gen import RnnlmGen
from models.attention_seq2seq import AttentionSeq2seq


def load_vocab(data_dir):
    """
    주어진 디렉토리에서 JSON 파일을 읽어 단어 사전 생성
    Args:
        data_dir (str): 데이터가 저장된 디렉토리 경로
    Returns:
        tuple: word_to_id, id_to_word
    """
    json_files = [os.path.join(data_dir, f)
                  for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {data_dir}")

    all_sentences = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_sentences.extend(data)
            elif isinstance(data, dict):
                def extract_strings(d):
                    if isinstance(d, dict):
                        for value in d.values():
                            yield from extract_strings(value)
                    elif isinstance(d, str):
                        yield d
                all_sentences.extend(extract_strings(data))
            else:
                raise ValueError(f"Unexpected JSON structure in file {file}")

    words = set(" ".join(all_sentences).split())
    word_to_id = {word: i for i, word in enumerate(sorted(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    return word_to_id, id_to_word


def generate_text(data_dir, model_path, start_text, max_length):
    import numpy as np

    # 단어 사전 로드
    print("Loading vocabulary...")
    word_to_id, id_to_word = load_vocab(data_dir)

    # 하이퍼파라미터 로드
    hyperparams_path = os.path.join(
        os.path.dirname(model_path), "hyperparams.json")
    with open(hyperparams_path, "r", encoding="utf-8") as f:
        hyperparams = json.load(f)

    # 모델 초기화
    vocab_size = hyperparams["vocab_size"]
    wordvec_size = hyperparams["wordvec_size"]
    hidden_size = hyperparams["hidden_size"]
    model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
    print("Loading model...")
    model.load_params(model_path)

    # 시작 텍스트를 ID로 변환 (문자 단위)
    start_ids = [word_to_id[char] for char in start_text if char in word_to_id]
    if not start_ids:
        raise ValueError(
            f"Start text '{start_text}' contains no known characters in the vocabulary."
        )

    # 입력 데이터 생성 (더미 입력)
    xs = np.array(start_ids).reshape(1, -1)

    # 인코더 출력 생성
    enc_hs = model.encoder.forward(xs)

    # 텍스트 생성
    print("Generating text...")
    word_ids = model.decoder.generate(
        enc_hs, start_id=start_ids[0], sample_size=max_length)
    txt = ''.join([id_to_word[i] for i in word_ids])
    txt = txt.replace('<eos>', '\n')
    print("Generated Text:\n")
    print(txt)
