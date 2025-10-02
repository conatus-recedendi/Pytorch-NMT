```sh
perl ./tokenizer.perl -l en < ../rewrite/
python wmt14.py
python limit_unk.py
python filter_long_sen.py
```

# train 용

```sh

perl ./tokenizer.perl -l en < ../rewrite/data/wmt/train14/en-de/train.en > ../rewrite/train.tokenized.en
perl ./tokenizer.perl -l de < ../rewrite/data/wmt/train14/en-de/train.de > ../rewrite/train.tokenized.de

cd ../rewrite
python limit_unk.py --train_en "./train.tokenized.en" --train_de "./train.tokenized.de" --out_en "train.50k.en" --out_de "train.50k.de" --ref_en "./train.tokenized.en" --ref_de "./train.tokenized.de"
python filter_long_sen.py --train_en "train.50k.en" --train_de "train.50k.de" --out_en "train.en" --out_de "train.de"
```

# wmttest14

```sh
perl ./tokenizer.perl -l en < ../rewrite/data/wmt/test14/en-de/src.en > ../rewrite/test.14.tokenized.en
perl ./tokenizer.perl -l de < ../rewrite/data/wmt/test14/en-de/ref.de > ../rewrite/test.14.tokenized.de

python limit_unk.py --train_en "./test.14.tokenized.en" --train_de "./test.14.tokenized.de" --out_en "test.14.50k.en" --out_de "test.14.50k.de" --ref_en "./test.14.tokenized.en"  --ref_de "./test.14.tokenized.de"

python filter_long_sen.py --train_en "test.14.50k.en" --train_de "test.14.50k.de" --out_en "test.14.en" --out_de "test.14.de"
```
