from datasets import load_dataset

# https://huggingface.co/datasets/wmt/wmt14
ds = load_dataset("wmt/wmt14", "de-en")


ds.save_to_disk("./data/wmt/train14/en-de")
