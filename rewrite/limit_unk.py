import collections


def build_vocab(file_path, vocab_size=50000):
    """주어진 파일에서 상위 vocab_size 단어의 집합을 리턴"""
    counter = collections.Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # counter.update(line.strip().split(b"\n"))
            counter.update(line.strip().split("\n"))
    most_common = [w for w, _ in counter.most_common(vocab_size)]
    return set(most_common)


def replace_with_unk(file_path, vocab, out_path):
    """파일을 읽어서 vocab에 없는 단어는 <unk>로 치환 후 저장"""
    with open(file_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            tokens = line.strip().split(b"\n")
            new_tokens = [tok if tok in vocab else "<unk>" for tok in tokens]

            fout.write(" ".join(new_tokens) + "\n")


if __name__ == "__main__":
    # 입력 파일 경로
    train_en = "./data/wmt/train14/en-de/train.en"
    train_de = "./data/wmt/train14/en-de/train.de"

    # 어휘 구축 (각각 별도)
    vocab_en = build_vocab(train_en, vocab_size=50000)
    vocab_de = build_vocab(train_de, vocab_size=50000)

    # 출력 파일 경로
    out_en = "train.50k.en"
    out_de = "train.50k.de"

    # <unk> 치환
    replace_with_unk(train_en, vocab_en, out_en)
    replace_with_unk(train_de, vocab_de, out_de)

    print("완료! ->", out_en, out_de)
