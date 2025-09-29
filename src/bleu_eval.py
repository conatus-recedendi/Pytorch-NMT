#!/usr/bin/env python3
"""
Python implementation of multi-bleu.perl
BLEU score evaluation script
"""

import sys
import os
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Tuple


def add_to_ref(file_path: str, ref_list: List[List[str]]) -> None:
    """Add reference sentences from file to reference list"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if len(ref_list) <= line_idx:
                    ref_list.append([])
                ref_list[line_idx].append(line)
    except FileNotFoundError:
        raise FileNotFoundError(f"Can't read {file_path}")


def get_ngrams(words: List[str], n: int) -> Dict[str, int]:
    """Extract n-grams from word list"""
    ngram_counts = defaultdict(int)
    for start in range(len(words) - n + 1):
        ngram = f"{n} " + " ".join(words[start : start + n])
        ngram_counts[ngram] += 1
    return dict(ngram_counts)


def my_log(x: float) -> float:
    """Safe logarithm function"""
    if x <= 0:
        return -9999999999
    return math.log(x)


def calculate_bleu(
    hypothesis_file=None, reference_stem: str = None, lowercase: bool = False
) -> None:
    """Calculate BLEU score"""

    # Load reference files
    ref_sentences = []
    ref_idx = 0

    # Check for reference files with different naming patterns
    if reference_stem:
        # Try reference0, reference1, ... pattern
        while os.path.exists(f"{reference_stem}{ref_idx}"):
            add_to_ref(f"{reference_stem}{ref_idx}", ref_sentences)
            ref_idx += 1

        # Try reference.ref0, reference.ref1, ... pattern
        if ref_idx == 0 and os.path.exists(f"{reference_stem}.ref0"):
            ref_idx = 0
            while os.path.exists(f"{reference_stem}.ref{ref_idx}"):
                add_to_ref(f"{reference_stem}.ref{ref_idx}", ref_sentences)
                ref_idx += 1

        # Try single reference file
        if ref_idx == 0 and os.path.exists(reference_stem):
            add_to_ref(reference_stem, ref_sentences)
            ref_idx = 1

    if ref_idx == 0:
        raise FileNotFoundError(
            f"ERROR: could not find reference file {reference_stem}"
        )

    # Initialize counters
    correct = [0] * 5  # Index 0 unused, 1-4 for n-grams
    total = [0] * 5  # Index 0 unused, 1-4 for n-grams
    length_translation = 0
    length_reference = 0

    # Process hypothesis sentences
    input_stream = hypothesis_file if hypothesis_file else sys.stdin
    sentence_idx = 0

    for line in input_stream:
        line = line.strip()
        if lowercase:
            line = line.lower()

        words = line.split()
        length_translation_this_sentence = len(words)

        # Find closest reference length
        closest_diff = 9999
        closest_length = 9999
        ref_ngram = {}

        if sentence_idx < len(ref_sentences):
            for reference in ref_sentences[sentence_idx]:
                if lowercase:
                    reference = reference.lower()
                ref_words = reference.split()
                ref_length = len(ref_words)
                diff = abs(length_translation_this_sentence - ref_length)

                if diff < closest_diff:
                    closest_diff = diff
                    closest_length = ref_length
                elif diff == closest_diff:
                    closest_length = min(closest_length, ref_length)

                # Collect n-grams for this reference
                for n in range(1, 5):
                    ref_ngram_n = get_ngrams(ref_words, n)
                    for ngram, count in ref_ngram_n.items():
                        if ngram not in ref_ngram or ref_ngram[ngram] < count:
                            ref_ngram[ngram] = count

        length_translation += length_translation_this_sentence
        length_reference += closest_length

        # Count n-gram matches
        for n in range(1, 5):
            t_ngram = get_ngrams(words, n)

            for ngram, count in t_ngram.items():
                # Extract n from ngram key
                n_value = int(ngram.split()[0])
                total[n_value] += count

                if ngram in ref_ngram:
                    if ref_ngram[ngram] >= count:
                        correct[n_value] += count
                    else:
                        correct[n_value] += ref_ngram[ngram]

        sentence_idx += 1

    # Calculate BLEU components
    bleu_scores = [0.0] * 5  # Index 0 unused

    for n in range(1, 5):
        if total[n] > 0:
            bleu_scores[n] = correct[n] / total[n]
        else:
            bleu_scores[n] = 0.0

    # Handle edge case
    if length_reference == 0:
        print("BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)")
        return

    # Calculate brevity penalty
    brevity_penalty = 1.0
    if length_translation < length_reference:
        brevity_penalty = math.exp(1 - length_reference / length_translation)

    # Calculate final BLEU score
    log_sum = sum(my_log(bleu_scores[n]) for n in range(1, 5))
    bleu = brevity_penalty * math.exp(log_sum / 4)

    # Print results
    print(
        f"BLEU = {100*bleu:.2f}, {100*bleu_scores[1]:.1f}/{100*bleu_scores[2]:.1f}/"
        f"{100*bleu_scores[3]:.1f}/{100*bleu_scores[4]:.1f} "
        f"(BP={brevity_penalty:.3f}, ratio={length_translation/length_reference:.3f}, "
        f"hyp_len={length_translation}, ref_len={length_reference})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BLEU score",
        usage="python bleu_eval.py [-lc] reference < hypothesis",
    )
    parser.add_argument(
        "-lc",
        "--lowercase",
        action="store_true",
        help="Convert to lowercase before evaluation",
    )
    parser.add_argument("reference", help="Reference file stem")
    parser.add_argument(
        "--hypothesis", type=str, help="Hypothesis file (default: stdin)"
    )

    args = parser.parse_args()

    # Handle hypothesis file
    hypothesis_file = None
    if args.hypothesis:
        hypothesis_file = open(args.hypothesis, "r", encoding="utf-8")

    try:
        calculate_bleu(hypothesis_file, args.reference, args.lowercase)
    finally:
        if hypothesis_file:
            hypothesis_file.close()


if __name__ == "__main__":
    main()
