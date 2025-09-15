import random


def filter_by_length(
    src_file, tgt_file, out_src, out_tgt, max_len=50, min_len=0, shuffle=True, seed=42
):
    src_lines = open(src_file, encoding="utf-8").read().strip().split("\n")
    tgt_lines = open(tgt_file, encoding="utf-8").read().strip().split("\n")

    assert len(src_lines) == len(tgt_lines), "Source/target 파일 길이가 다릅니다!"

    filtered_pairs = []
    for s, t in zip(src_lines, tgt_lines):
        if (
            len(s.split()) <= max_len
            and len(t.split()) <= max_len
            and len(s.split()) > min_len
            and len(t.split()) > min_len
        ):
            filtered_pairs.append((s, t))

    print(f"Before filtering: {len(src_lines)} pairs")
    print(f"After filtering: {len(filtered_pairs)} pairs")

    if shuffle:
        random.seed(seed)
        random.shuffle(filtered_pairs)

    with open(out_src, "w", encoding="utf-8") as fsrc, open(
        out_tgt, "w", encoding="utf-8"
    ) as ftgt:
        for s, t in filtered_pairs:
            fsrc.write(s + "\n")
            ftgt.write(t + "\n")


if __name__ == "__main__":
    filter_by_length(
        src_file="./data/wmt/train14/en-de/train.en",
        tgt_file="./data/wmt/train14/en-de/train.de",
        out_src="train.len50.en",
        out_tgt="train.len50.de",
        max_len=50,
        min_len=0,
        shuffle=True,
    )
