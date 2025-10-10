# bleu_simple.py
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

ref = [line.strip().split() for line in open("ref.txt", encoding="utf-8")]
hyp = [line.strip().split() for line in open("hyp.txt", encoding="utf-8")]

refs = [[r] for r in ref]  # 각 문장마다 참조 리스트
score = corpus_bleu(refs, hyp, smoothing_function=SmoothingFunction().method1)

print(f"BLEU: {score * 100:.2f}")
