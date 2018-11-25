import argparse
import etl
import helpers
import torch
from attention_decoder import AttentionDecoderRNN
from encoder import EncoderRNN
from language import Language


# Parse argument for input sentence
parser = argparse.ArgumentParser()
parser.add_argument('--attn_model', type=str, help='attention type: dot, general, concat')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--n_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--language', type=str, help='specific which language.')
parser.add_argument('--input', type=str, help='src -> tgt')
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--seed', type=str, help='random seed')
args = parser.parse_args()
helpers.validate_language_params(args.language)

input_lang, output_lang, pairs = etl.prepare_data(args.language)

torch.random.manual_seed(args.seed)

device = torch.device(args.device)

print('input: %s' % args.input)

# Initialize models
encoder = EncoderRNN(
    input_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.n_layers,
    args.dropout
)

decoder = AttentionDecoderRNN(
    output_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.attn_model,
    args.n_layers,
    args.dropout
)

# Load model parameters
encoder.load_state_dict(torch.load('./data/encoder_params_{}'.format(args.language)))
decoder.load_state_dict(torch.load('./data/decoder_params_{}'.format(args.language)))
decoder.attention.load_state_dict(torch.load('./data/attention_params_{}'.format(args.language)))

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)


def evaluate(sentence, max_length=10):
    input = etl.tensor_from_sentence(input_lang, sentence, device)
    input_length = input.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden(device)
    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[Language.sos_token]]).to(device)  # SOS
    decoder_context = torch.zeros(1, decoder.hidden_size).to(device)

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                     decoder_context,
                                                                                     decoder_hidden,
                                                                                     encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi.item()
        if ni == Language.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = topi

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

sentence = helpers.normalize_string(args.input)
output_words, decoder_attn = evaluate(sentence)
final_words = []
for word in output_words:
    if word in ['<EOS>', '<PAD>']:
        continue
    elif word == '<SOS>':
        break
    final_words.append(word)
output_sentence = ' '.join(final_words)
print(output_sentence)
