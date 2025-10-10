# bleu_simple.py
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

ref = [line.strip().split() for line in open("../rewrite/test.14.de", encoding="utf-8")]
hyp = [
    line.strip().split() for line in open("../test.14.hypothesis.de", encoding="utf-8")
]

refs = [[r] for r in ref]  # 각 문장마다 참조 리스트
score = corpus_bleu(refs, hyp, smoothing_function=SmoothingFunction().method1)

print(f"BLEU: {score * 100:.2f}")
