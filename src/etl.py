import helpers
import torch
from language import Language

"""
Data Extraction
"""

max_length = 20


def filter_pair(p):
    is_good_length = (
        len(p[0].split(" ")) < max_length and len(p[1].split(" ")) < max_length
    )
    return is_good_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang_name):

    # Read and filter sentences
    input_lang, output_lang, pairs = read_languages(lang_name)
    pairs = filter_pairs(pairs)

    # Index words
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


def read_languages(lang):

    # Read and parse the text file
    doc = open("./rewrite/train.len50.%s" % lang, "rb")
    lines = doc.read().strip().split(b"\n")
    lines = [l.decode("utf-8", errors="strict") for l in lines]

    doc_en = open("./rewrite/train.len50.en", "rb")
    lines_en = doc_en.read().strip().split(b"\n")
    lines_en = [l.decode("utf-8", errors="strict") for l in lines_en]
    print("loaded")

    pairs = [[s, t] for s, t in zip(lines_en, lines)]
    print("read %s sentence pairs" % len(pairs))

    # Transform the data and initialize language instances
    # pairs = [[helpers.normalize_string(s) for s in l.split("\t")] for l in lines]

    input_lang = Language("eng")
    output_lang = Language(lang)

    return input_lang, output_lang, pairs


"""
Data Transformation
"""


# Returns a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensor_from_sentence(lang, sentence, device="cpu"):
    print(sentence)
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(Language.eos_token)
    tensor = torch.LongTensor(indexes).view(-1, 1).to(device)
    return tensor


def tensor_from_pair(pair_batch, input_lang, output_lang, device="cpu"):
    # empty e tensor
    batch_input: list = []
    batch_target = []
    for pair in pair_batch:
        input = tensor_from_sentence(input_lang, pair[0], device)
        target = tensor_from_sentence(output_lang, pair[1], device)
        batch_input.append(input)
        batch_target.append(target)

    return batch_input, batch_target
