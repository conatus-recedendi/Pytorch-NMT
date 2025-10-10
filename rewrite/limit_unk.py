import collections
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse
import re
import unicodedata


# Turns a unicode string to plain ASCII (http://stackoverflow.com/a/518232/2809427)
def unicode_to_ascii(s):
    chars = [
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" or c == "<" or c == ">"
    ]
    char_list = "".join(chars)
    return char_list


def normalize_string(s):
    s = s.lower().strip()
    # <unk> 은 보존
    s = re.sub(rb"([.!?])", rb" \1", s)  # 구두점 앞에 공백 추가
    s = re.sub(rb"[^a-zA-Z.!?,()\s]+", rb"", s)  # 영문자, 구두점, 공백 외 제거
    s = re.sub(rb"\s{2,}", rb" ", s)  # 여러 공백 -> 하나의 공백
    return s


def build_vocab(file_path, vocab_size=50000):
    """주어진 파일에서 상위 vocab_size 단어의 집합을 리턴"""
    counter = collections.Counter()
    with open(file_path, "rb") as f:
        data = (
            f.read().decode("utf-8", errors="strict").encode("utf-8", errors="strict")
        )

        # print(data[:100])
        idx = 0
        for line in data.split(b"\n"):
            idx += 1
            tokens = line.strip().split(b" ")
            tokens = [
                normalize_string(tok).strip() for tok in tokens if tok and tok.strip()
            ]
            # 빈 토큰과 공백만 있는 토큰 제거
            # tokens = [tok for tok in tokens if tok and tok.strip()]
            counter.update(tokens)
            # counter.update(line.strip().split(b" "))

    most_common = [w for w, _ in counter.most_common(vocab_size)]
    most_common.sort()
    print(most_common)
    return set(most_common)


def replace_with_unk(file_path, vocab, out_path):
    """파일을 읽어서 vocab에 없는 단어는 <UNK>로 치환 후 저장"""
    with open(file_path, "rb") as fin, open(out_path, "wb") as fout:
        data = (
            fin.read().decode("utf-8", errors="strict").encode("utf-8", errors="strict")
        )
        for line in data.split(b"\n"):

            # line = re.sub(r"\s{2,}", r" ", line)
            # remove empty token bcz it is biniary
            # line = re.sub(r"^\s+|\s+$", r"", line)

            tokens = line.strip().split(b" ")
            tokens = [
                normalize_string(tok).strip() for tok in tokens if tok and tok.strip()
            ]

            new_tokens = [tok if tok in vocab else b"<unk>" for tok in tokens]
            # print(new_tokens)

            fout.write(b" ".join(new_tokens) + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_en", type=str, default="./data/wmt/train14/en-de/train.en"
    )
    parser.add_argument(
        "--train_de", type=str, default="./data/wmt/train14/en-de/train.de"
    )
    parser.add_argument("--out_en", type=str, default="train.50k.en")
    parser.add_argument("--out_de", type=str, default="train.50k.de")
    parser.add_argument("--ref_en", type=str, default="train.50k.en")
    parser.add_argument("--ref_de", type=str, default="train.50k.de")
    args = parser.parse_args()

    # 입력 파일 경로
    train_en = args.train_en
    train_de = args.train_de

    # 어휘 구축 (각각 별도)
    # vocab_en = build_vocab(train_en, vocab_size=50000)
    vocab_en = build_vocab(args.ref_en, vocab_size=50000)

    print(list(vocab_en)[:10])
    # vocab_de = build_vocab(train_de, vocab_size=50000)
    vocab_de = build_vocab(args.ref_de, vocab_size=50000)

    print(len(vocab_en), len(vocab_de))

    # 출력 파일 경로
    out_en = args.out_en
    out_de = args.out_de

    # <unk> 치환
    replace_with_unk(train_en, vocab_en, out_en)
    replace_with_unk(train_de, vocab_de, out_de)

    print("완료! ->", out_en, out_de)
