import random
from itertools import zip_longest
import unicodedata
import re


# Turns a unicode string to plain ASCII (http://stackoverflow.com/a/518232/2809427)
def unicode_to_ascii(s):
    chars = [
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    ]
    char_list = "".join(chars)
    return char_list


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_normalized_lines(path):
    # 개행 정규화: CRLF/CR -> LF, 개행만 제거하고 내용 공백은 보존
    with open(path, "rb") as f:
        data = (
            f.read().decode("utf-8", errors="strict").encode("utf-8", errors="strict")
        )
    # data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # 마지막 개행으로 끝나면 마지막에 b"" 요소가 생김 -> 실제 '빈 줄'로 취급
    # 만약 한 라인에 \n이 여러개 있으면 -> 하나로 간주함
    # \r\n이 아니 \n으로만 split
    # lines = data.split(b"\n")
    lines = data.split(b"\n")
    # bytes -> str (UTF-8)
    return [ln.decode("utf-8", errors="strict") for ln in lines]


def filter_by_length_strict(
    src_file, tgt_file, out_src, out_tgt, max_len=50, min_len=0, shuffle=True, seed=42
):
    src_lines = read_normalized_lines(src_file)
    tgt_lines = read_normalized_lines(tgt_file)

    # 길이 다르면 어디서 깨졌는지 위치/상태 출력
    if len(src_lines) != len(tgt_lines):
        from itertools import zip_longest

        for i, (s, t) in enumerate(zip_longest(src_lines, tgt_lines, fillvalue=None)):
            if s is None or t is None:
                raise AssertionError(
                    f"Source/target lengths differ at index {i}: "
                    f"{len(src_lines)} vs {len(tgt_lines)}"
                )
        # 혹시 여기 안 걸리면 아래 assert에서 걸림
    assert len(src_lines) == len(
        tgt_lines
    ), f"Len mismatch: {len(src_lines)} vs {len(tgt_lines)}"

    filtered_pairs = []
    dropped_mismatch = 0
    dropped_empty_pair = 0

    for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
        # 양쪽 모두 완전 빈 줄이면 같이 스킵 (쌍 정합 유지)
        if s == "" and t == "":
            dropped_empty_pair += 1
            continue
        # 한쪽만 빈 줄이면 데이터 자체가 깨진 것이므로 에러로 표시
        if (s == "" and t != "") or (s != "" and t == ""):
            dropped_mismatch += 1
            # 필요 시 raise로 바꾸세요
            continue

        if (min_len < len(s.split()) <= max_len) and (
            min_len < len(t.split()) <= max_len
        ):
            filtered_pairs.append((s, t))

    print(f"Before filtering (incl. possible empty last line): {len(src_lines)} pairs")
    print(f"Dropped empty pairs: {dropped_empty_pair}")
    print(f"Dropped mismatched empties: {dropped_mismatch}")
    print(f"After filtering: {len(filtered_pairs)} pairs")

    if shuffle:
        random.seed(seed)
        random.shuffle(filtered_pairs)

    with open(out_src, "w", encoding="utf-8", newline="\n") as fsrc, open(
        out_tgt, "w", encoding="utf-8", newline="\n"
    ) as ftgt:
        for s, t in filtered_pairs:
            fsrc.write(normalize_string(s) + "\n")
            ftgt.write(normalize_string(t) + "\n")


if __name__ == "__main__":
    filter_by_length_strict(
        src_file="train.50k.en",
        tgt_file="train.50k.de",
        out_src="train.len50.en",
        out_tgt="train.len50.de",
        max_len=50,
        min_len=0,
        shuffle=True,
    )
