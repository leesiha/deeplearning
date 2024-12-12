# coding: utf-8
import sys
import os
import json
from models.rnnlm_gen import RnnlmGen


def load_vocab(data_dir):
    """
    주어진 디렉토리에서 JSON 파일을 읽어 단어 사전 생성
    Args:
        data_dir (str): 데이터가 저장된 디렉토리 경로
    Returns:
        tuple: word_to_id, id_to_word
    """
    # JSON 파일 경로 탐색
    json_files = [os.path.join(data_dir, f)
                  for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory {data_dir}")

    # 데이터 읽기 및 단어 사전 생성
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

    # 단어 사전 생성
    words = set(" ".join(all_sentences).split())
    word_to_id = {word: i for i, word in enumerate(sorted(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    return word_to_id, id_to_word


def generate_text(data_dir, model_path, start_word, skip_words):
    """
    RNN 언어 모델을 사용하여 텍스트 생성
    Args:
        data_dir (str): 데이터 디렉토리 경로
        model_path (str): 모델 가중치 파일 경로
        start_word (str): 텍스트 생성 시작 단어
        skip_words (list): 건너뛸 단어 리스트
    """
    # 단어 사전 로드
    print("Loading vocabulary...")
    word_to_id, id_to_word = load_vocab(data_dir)

    # 모델 로드
    print("Loading model...")
    model = RnnlmGen()
    model.load_params(model_path)

    # start_word와 skip_words 처리
    if start_word not in word_to_id:
        raise ValueError(f"Start word '{start_word}' not in vocabulary.")
    start_id = word_to_id[start_word]
    skip_ids = [word_to_id[word] for word in skip_words if word in word_to_id]

    # 텍스트 생성
    print("Generating text...")
    word_ids = model.generate(start_id, skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print("Generated Text:\n")
    print(txt)


if __name__ == "__main__":
    import argparse

    # ArgumentParser로 명령줄 인자 처리
    parser = argparse.ArgumentParser(
        description="Generate text using a trained RNN language model.")
    parser.add_argument("--data_dir", type=str, default="./data/Training/01.원천데이터/TS_1.발화단위평가_기술_과학", help="Path to the data directory")
    parser.add_argument("--model_path", type=str, default="../ch06/Rnnlm.pkl", help="Path to the model parameter file")
    parser.add_argument("--start_word", type=str, default="you", help="Starting word for text generation")
    parser.add_argument("--skip_words", nargs='*', default=['N', '<unk>', '$'], help="List of words to skip during text generation")
    args = parser.parse_args()

    # 텍스트 생성 함수 호출
    generate_text(
        data_dir=args.data_dir,
        model_path=args.model_path,
        start_word=args.start_word,
        skip_words=args.skip_words
    )
