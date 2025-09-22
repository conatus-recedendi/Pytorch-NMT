```sh
python wmt14.py
python limit_unk.py
python filter_long_sen.py
```

```sh
python limit_unk.py --train_en "./data/wmt/test14/en-de/src.en" --train_de "./data/wmt/test14/en-de/ref.de" --out_en "test.14.50k.en" --out_de "test.14.50k.de"

python filter_long_sen.py --train_en "test.14.50k.en" --train_de "test.14.50k.de" --out_en "test.14.en" --out_de "test.14.de"
```
