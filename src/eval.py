import argparse
import etl
import helpers
import torch
from attention_decoder import AttentionDecoderRNN
from topk_decode import TopKDecode
from encoder import EncoderRNN
from language import Language
from beam import Beam


# Parse argument for input sentence
parser = argparse.ArgumentParser()
parser.add_argument('--attn_model', type=str, help='attention type: dot, general, concat')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--n_layers', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--language', type=str, help='specific which language.')
parser.add_argument('--input', type=str, help='src -> tgt')
parser.add_argument('--max_len', type=int)
parser.add_argument('--beam_size', type=int)
parser.add_argument('--batch_size', type=int)
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


def evaluate(sentence, max_len=10):
    input = etl.tensor_from_sentence(input_lang, sentence, device)
    input_length = input.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden(device)
    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)

    # Create starting vectors for decoder
    decoder_context = torch.zeros(1, 1, decoder.hidden_size).to(device)

    decoder_hidden = encoder_hidden

    topk_decoder = TopKDecode(
        decoder,
        decoder.hidden_size,
        args.beam_size,
        output_lang.n_words,
        Language.sos_token,
        Language.eos_token,
        device
    )
    topk_decoder = topk_decoder.to(device)


    decoder_outputs, _, metadata = topk_decoder(
        decoder_context,
        decoder_hidden,
        encoder_outputs,
        args.max_len,
        args.batch_size,
    )

    beam_words = torch.stack(metadata['topk_sequence'], dim=0)
    #  print(beam_words.shape)
    beam_words = beam_words.squeeze(3).squeeze(1).transpose(0, 1)
    beam_length = metadata['topk_length']
    print_sentence(beam_words, beam_length[0], 'beam')

    """
    beam_words, _, _= beam_decode(
        decoder_context,
        decoder_hidden,
        encoder_outputs,
        max_len,
        beam_size=5
    )
    # [batch_size, beam_size, max_len] -> [beam_size, max_len] because we
    # batch_size if 1.
    beam_words = beam_words[0]
    #  print(beam_words)
    print_sentence(beam_words, 'beam')

    """
    greedy_words, greedy_attention = greedy_decode(
        decoder_context,
        decoder_hidden,
        encoder_outputs,
        max_len
    )
    print_sentence(greedy_words)


def greedy_decode(decoder_context,
                  decoder_hidden,
                  encoder_outputs,
                  max_len):
    # Run through decoder
    decoded_words = []
    decoder_attentions = torch.zeros(max_len, max_len)
    decoder_input = torch.LongTensor([[Language.sos_token]]).to(device)  # SOS
    for di in range(max_len):
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

    return decoded_words, decoder_attentions[:di + 1, :encoder_outputs.size(0)]

def beam_decode(decoder_context,
                decoder_hidden,
                encoder_outputs,
                max_len,
                beam_size=5):
    batch_size = args.beam_size
    vocab_size = output_lang.n_words
    # [1, batch_size x beam_size]
    decoder_input = torch.ones(batch_size * beam_size, dtype=torch.long, device=device) * Language.sos_token

    # [num_layers, batch_size x beam_size, hidden_size]
    decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
    decoder_context = decoder_context.repeat(1, beam_size, 1)

    encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

    # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
    batch_position = torch.arange(0, batch_size, dtype=torch.long, device=device) * beam_size

    score = torch.ones(batch_size * beam_size, device=device) * -float('inf')
    score.index_fill_(0, torch.arange(0, batch_size, dtype=torch.long, device=device) * beam_size, 0.0)

    # Initialize Beam that stores decisions for backtracking
    beam = Beam(
        batch_size,
        beam_size,
        max_len,
        batch_position,
        Language.eos_token
    )

    for i in range(max_len):
        decoder_output, decoder_context, decoder_hidden, _ = decoder(decoder_input,
                                                                    decoder_context,
                                                                    decoder_hidden,
                                                                    encoder_outputs)
        # output: [1, batch_size * beam_size, vocab_size]
        # -> [batch_size * beam_size, vocab_size]
        log_prob = decoder_output

        # score: [batch_size * beam_size, vocab_size]
        score = score.view(-1, 1) + log_prob

        # score [batch_size, beam_size]
        score, top_k_idx = score.view(batch_size, -1).topk(beam_size, dim=1)

        # decoder_input: [batch_size x beam_size]
        decoder_input = (top_k_idx % vocab_size).view(-1)

        # beam_idx: [batch_size, beam_size]
        beam_idx = top_k_idx / vocab_size  # [batch_size, beam_size]

        # top_k_pointer: [batch_size * beam_size]
        top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

        # [num_layers, batch_size * beam_size, hidden_size]
        decoder_hidden = decoder_hidden.index_select(1, top_k_pointer)
        decoder_context = decoder_context.index_select(1, top_k_pointer)

        # Update sequence scores at beam
        beam.update(score.clone(), top_k_pointer, decoder_input)

        # Erase scores for EOS so that they are not expanded
        # [batch_size, beam_size]
        eos_idx = decoder_input.data.eq(Language.eos_token).view(batch_size, beam_size)

        if eos_idx.nonzero().dim() > 0:
            score.data.masked_fill_(eos_idx, -float('inf'))

    prediction, final_score, length = beam.backtrack()
    return prediction, final_score, length


def assemble_sentence(words):
    final_words = list()
    for word in words:
        if word in ['<SOS>', '<PAD>']:
            continue
        elif word == '<EOS>':
            break
        final_words.append(word)
    sentence = ' '.join(final_words)
    return sentence

def print_sentence(words, lengths=None, mode='greedy'):
    if mode == 'greedy':
        print('greedy > %s' % assemble_sentence(words))
    elif mode == 'beam':
        for i, (length, ids) in enumerate(zip(lengths, words.tolist())):
            cur_words = [output_lang.index2word[id] for id in ids[:length]]
            sentence = assemble_sentence(cur_words)
            print('beam %d > %s' % (i, sentence))

input_sentence = helpers.normalize_string(args.input)
evaluate(input_sentence, args.max_len)

