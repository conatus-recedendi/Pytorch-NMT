from datasets import load_dataset

ds = load_dataset("wmt/wmt14", "de-en")


ds.save_to_disk("./data/wmt/train14/en-de")
