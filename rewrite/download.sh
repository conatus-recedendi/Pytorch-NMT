#!/bin/bash
set -e

mkdir -p data/wmt/{test13,test14,test15}/{en-de,de-en}

# EN→DE
sacrebleu -t wmt13 -l en-de --echo ref > data/wmt/test13/en-de/ref.de
sacrebleu -t wmt13 -l en-de --echo src > data/wmt/test13/en-de/src.en
sacrebleu -t wmt14 -l en-de --echo ref > data/wmt/test14/en-de/ref.de
sacrebleu -t wmt14 -l en-de --echo src > data/wmt/test14/en-de/src.en
sacrebleu -t wmt15 -l en-de --echo ref > data/wmt/test15/en-de/ref.de
sacrebleu -t wmt15 -l en-de --echo src > data/wmt/test15/en-de/src.en

# DE→EN
sacrebleu -t wmt13 -l de-en --echo ref > data/wmt/test13/de-en/ref.en
sacrebleu -t wmt13 -l de-en --echo src > data/wmt/test13/de-en/src.de
sacrebleu -t wmt14 -l de-en --echo ref > data/wmt/test14/de-en/ref.en
sacrebleu -t wmt14 -l de-en --echo src > data/wmt/test14/de-en/src.de
sacrebleu -t wmt15 -l de-en --echo ref > data/wmt/test15/de-en/ref.en
sacrebleu -t wmt15 -l de-en --echo src > data/wmt/test15/de-en/src.de


# 저장할 디렉토리
mkdir -p data/wmt/train14/en-de

# 1) Europarl v7
if [ ! -f data/wmt/train14/en-de/de-en.tgz ]; then
    echo "Downloading Europarl v7..."
    wget -P data/wmt/train14/en-de http://www.statmt.org/europarl/v7/de-en.tgz
fi
tar -xvf data/wmt/train14/en-de/de-en.tgz -C data/wmt/train14/en-de

# 2) News Commentary v9
# https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
if [ ! -f data/wmt/train14/en-de/training-parallel-nc-v9.tgz ]; then
    echo "Downloading News Commentary v9..."
    wget -P data/wmt/train14/en-de https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
fi
tar -xvf data/wmt/train14/en-de/training-parallel-nc-v9.tgz -C data/wmt/train14/en-de
# 3) Common Crawl
# https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
if [ ! -f data/wmt/train14/en-de/training-parallel-commoncrawl.tgz ]; then
    echo "Downloading Common Crawl..."
    wget -P data/wmt/train14/en-de https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
fi
tar -xvf data/wmt/train14/en-de/training-parallel-commoncrawl.tgz -C data/wmt/train14/en-de
# 4) 정리: 평행 문장 파일 추출
# 예: europarl-v7 has files like: europarl-v7.de-en.en, europarl-v7.de-en.de
# news-commentary v9: similar
# commoncrawl: similar

# (Optional) 조합하여 하나의 병렬 파일로 합치기
cat data/wmt/train14/en-de/europarl-v7.de-en.en data/wmt/train14/en-de/training/news-commentary-v9.de-en.en data/wmt/train14/en-de/commoncrawl.de-en.en > data/wmt/train14/en-de/train.en
cat data/wmt/train14/en-de/europarl-v7.de-en.de data/wmt/train14/en-de/training/news-commentary-v9.de-en.de data/wmt/train14/en-de/commoncrawl.de-en.de > data/wmt/train14/en-de/train.de

echo "Downloaded and concatenated WMT14 EN-DE training data."
