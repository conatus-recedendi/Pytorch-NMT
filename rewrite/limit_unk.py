import collections
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import argparse
import re


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
            # if idx < 100:
            #     print(line.strip().split(b" "))
            counter.update(line.strip().split(b" "))

    most_common = [w for w, _ in counter.most_common(vocab_size)]
    return set(most_common)


def replace_with_unk(file_path, vocab, out_path):
    """파일을 읽어서 vocab에 없는 단어는 <UNK>로 치환 후 저장"""
    with open(file_path, "rb") as fin, open(out_path, "wb") as fout:
        data = (
            fin.read().decode("utf-8", errors="strict").encode("utf-8", errors="strict")
        )
        for line in data.split(b"\n"):

            tokens = line.strip().split(b" ")
            tokens = re.sub(r"\s{2,}", " ", tokens)
            # print(tokens)
            new_tokens = [tok if tok in vocab else b"<UNK>" for tok in tokens]
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
