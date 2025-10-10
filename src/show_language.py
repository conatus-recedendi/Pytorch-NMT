import etl


input_lang, output_lang, pairs = etl.prepare_data("de", is_test=True)


print(input_lang.n_words, output_lang.n_words, len(pairs))

print(output_lang.word2index)
