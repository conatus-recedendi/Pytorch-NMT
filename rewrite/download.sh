mkdir -p data/wmt/{dev,test14,test15}/{en-de,de-en}

# EN→DE
sacrebleu -t wmt13 -l en-de --echo ref > data/wmt/dev/en-de/ref.de
sacrebleu -t wmt13 -l en-de --echo src > data/wmt/dev/en-de/src.en
sacrebleu -t wmt14 -l en-de --echo ref > data/wmt/test14/en-de/ref.de
sacrebleu -t wmt14 -l en-de --echo src > data/wmt/test14/en-de/src.en
sacrebleu -t wmt15 -l en-de --echo ref > data/wmt/test15/en-de/ref.de
sacrebleu -t wmt15 -l en-de --echo src > data/wmt/test15/en-de/src.en

# DE→EN
sacrebleu -t wmt13 -l de-en --echo ref > data/wmt/dev/de-en/ref.en
sacrebleu -t wmt13 -l de-en --echo src > data/wmt/dev/de-en/src.de
sacrebleu -t wmt14 -l de-en --echo ref > data/wmt/test14/de-en/ref.en
sacrebleu -t wmt14 -l de-en --echo src > data/wmt/test14/de-en/src.de
sacrebleu -t wmt15 -l de-en --echo ref > data/wmt/test15/de-en/ref.en
sacrebleu -t wmt15 -l de-en --echo src > data/wmt/test15/de-en/src.de
