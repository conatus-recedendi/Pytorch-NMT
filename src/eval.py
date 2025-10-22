import argparse
import etl
import helpers
import torch
from attention_decoder import AttentionDecoderRNN
from topk_decode import TopKDecode
from encoder import EncoderRNN
from language import Language
from beam import Beam
from torch import nn
import torch.nn.functional as F
import math

# import numpy as np


def pad_sequences_pre(sequences, maxlen, padding_value=0):
    """
    pad_sequences와 동일하지만 앞쪽(pre)에 padding을 붙입니다.
    """
    padded = []
    for seq in sequences:
        seq = torch.tensor(seq)
        if len(seq) < maxlen:
            pad_width = maxlen - len(seq)
            padded_seq = torch.nn.functional.pad(
                seq, (pad_width, 0), mode="constant", value=padding_value
            )
        else:
            padded_seq = seq[-maxlen:]  # 길이 초과 시 뒤쪽 자름
        padded.append(padded_seq)
    return torch.stack(padded)


# Parse argument for input sentence
parser = argparse.ArgumentParser()
parser.add_argument(
    "--attn_model", type=str, help="attention type: dot, general, concat"
)
parser.add_argument("--embedding_size", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--dropout", type=float)
parser.add_argument("--language", type=str, help="specific which language.")
parser.add_argument(
    "--input_file", type=str, help="input file path with source sentences"
)
parser.add_argument(
    "--output_file", type=str, help="output file path for translations (optional)"
)
parser.add_argument("--max_len", type=int)
parser.add_argument("--beam_size", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device", type=str, help="cpu or cuda")
parser.add_argument("--seed", type=str, help="random seed")
parser.add_argument("--local", type=str, help="local-m, local-p, None", default=None)
parser.add_argument(
    "--input_ref_file", type=str, help="input reference file path with target sentences"
)
parser.add_argument(
    "--reverse", type=bool, help="reverse source sentence", default=False
)
parser.add_argument("--input_forward", type=bool, help="input feeding", default=False)
parser.add_argument("--clip_forward", type=float, default=50.0)
parser.add_argument("--clip_backward", type=float, default=1000.0)
args = parser.parse_args()
# helpers.validate_language_params(args.language)

input_lang, output_lang, pairs = etl.prepare_data(args.language, is_test=True)

torch.random.manual_seed(args.seed)

device = torch.device(args.device)

print("input file: %s" % args.input_file)

# Initialize models
encoder = EncoderRNN(
    args.batch_size,
    input_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.n_layers,
    args.dropout,
    args.clip_forward,
    args.clip_backward,
)

print(
    "encoder: n_words=%d, embedding_size=%d, hidden_size=%d, n_layers=%d, dropout=%.2f, reverse=%s, clip_forward=%s, clip_backward=%s"
    % (
        input_lang.n_words,
        args.embedding_size,
        args.hidden_size,
        args.n_layers,
        args.dropout,
        args.reverse,
        args.clip_forward,
        args.clip_backward,
    )
)

decoder = AttentionDecoderRNN(
    args.batch_size,
    output_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.attn_model,
    args.n_layers,
    args.dropout,
    args.local,
    args.clip_forward,
    args.clip_backward,
    args.input_forward,
)
encoder.eval()
decoder.eval()

id = "id=12_attn=%s,local=%s,dropout=d%.2f,epoch=12" % (
    args.attn_model,
    args.local if args.local else "global",
    args.dropout,
)
# Load model parameters with compatibility handling
try:
    encoder.load_state_dict(torch.load("./data/encoder_model_{}".format(id)))
    decoder.load_state_dict(torch.load("./data/decoder_model_{}".format(id)))
    print("Models loaded successfully")
except RuntimeError as e:
    print(f"Error loading models: {e}")
    print("Attempting to load with key mapping...")

    # Load encoder with key mapping
    encoder_state = torch.load("./data/encoder_model_{}".format(id))
    encoder_mapped_state = {}
    for key, value in encoder_state.items():
        encoder_mapped_state[key] = value

    encoder.load_state_dict(encoder_mapped_state, strict=False)

    # Load decoder with key mapping
    decoder_state = torch.load("./data/decoder_model_{}".format(id))
    decoder_mapped_state = {}
    for key, value in decoder_state.items():
        if key.startswith("lstm.lstm."):
            # lstm.lstm.weight_ih_l0 -> lstm.lstm_layers.0.weight_ih_l0
            # lstm.lstm.weight_ih_l1 -> lstm.lstm_layers.1.weight_ih_l0
            # 키에서 레이어 번호 추출 (마지막 _lX 부분에서)
            import re

            match = re.search(r"_l(\d+)$", key)
            if match:
                layer_num = match.group(1)
                # lstm.lstm.weight_ih_l0 -> lstm_layers.0.weight_ih_l0 형태로 변환
                base_key = key.replace("lstm.lstm.", "").replace(
                    f"_l{layer_num}", "_l0"
                )
                new_key = f"lstm.lstm_layers.{layer_num}.{base_key}"
            else:
                # 레이어 번호가 없는 경우 그대로 변환
                new_key = key.replace("lstm.lstm.", "lstm.lstm_layers.0.")

            decoder_mapped_state[new_key] = value
        else:
            decoder_mapped_state[key] = value

    print(decoder_mapped_state.keys())
    decoder.load_state_dict(decoder_mapped_state, strict=False)
    print("Models loaded with key mapping")

# Only load attention weights if not base model
if args.attn_model != "base":
    try:
        decoder.attention.load_state_dict(
            torch.load("./data/attention_model_{}".format(id))
        )
        decoder.attention.eval()

        print("Attention weights loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load attention weights: {e}")

# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)


def evaluate_sentence(sentence, ref_sentence, max_len=10):
    input = etl.tensor_from_sentence(
        input_lang, sentence, device, is_src=True, is_reverse=args.reverse
    )
    target = etl.tensor_from_sentence(
        output_lang, ref_sentence, device, is_src=False, is_reverse=False
    )
    ref_pad_cnt = (target == Language.pad_token).sum().item()

    input_length = input.size()[0]

    # Run through encoder
    input = input.view(1, -1)  # [1, len]
    # input = pad_sequences_pre(input, maxlen=50, padding_value=2)  # PAD token
    encoder_hidden = encoder.init_hidden(device, 1)

    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)

    # Create starting vectors for decoder
    decoder_context = torch.zeros(1, 1, decoder.hidden_size).to(device)

    decoder_hidden = encoder_hidden

    # Beam search decode
    # topk_decoder = TopKDecode(
    #     decoder,
    #     decoder.hidden_size,
    #     args.beam_size,
    #     output_lang.n_words,
    #     Language.sos_token,
    #     Language.eos_token,
    #     device,
    # )
    # topk_decoder = topk_decoder.to(device)

    # decoder_outputs, _, metadata = topk_decoder(
    #     decoder_context,
    #     decoder_hidden,
    #     encoder_outputs,
    #     args.max_len,
    #     args.batch_size,
    #     targets=target,  # Pass targets for loss calculation
    # )

    # beam_words = torch.stack(metadata["topk_sequence"], dim=0)
    # beam_words = beam_words.squeeze(3).squeeze(1).transpose(0, 1)
    # beam_length = metadata["topk_length"]

    # # Get best beam translation and loss
    # best_beam_ids = beam_words[0][: beam_length[0][0]]
    # best_beam_words = [output_lang.index2word[id] for id in best_beam_ids.tolist()]
    # best_beam_sentence = assemble_sentence(best_beam_words)
    # beam_loss = metadata.get("loss", float("inf"))

    # # Also get greedy translation for comparison
    greedy_words, greedy_attention, greedy_loss, _ = greedy_decode(
        decoder_context, decoder_hidden, encoder_outputs, max_len, target
    )
    _, _, ppl_loss, valid_token = greedy_decode(
        decoder_context,
        decoder_hidden,
        encoder_outputs,
        max_len,
        target,
        is_teaching_force=True,
    )
    greedy_sentence = assemble_sentence(greedy_words)

    # return best_beam_sentence, greedy_sentence, ppl_loss, beam_loss, ref_pad_cnt
    return None, greedy_sentence, ppl_loss, None, ref_pad_cnt, valid_token


def evaluate_file(input_file_path, input_ref_path, output_file_path=None, max_len=10):
    """Evaluate sentences from input file"""
    results = []

    # Read input sentences
    with open(input_file_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    with open(input_ref_path, "r", encoding="utf-8") as f:
        ref_sentences = [line.strip() for line in f if line.strip()]

    # Process each sentence

    idx = 0
    loss = 0
    total_token = 0
    beam_loss_total = 0
    ref_pad_cnt_total = 0
    for sentence in sentences:
        ref_sentence = ref_sentences[idx] if idx < len(ref_sentences) else None

        # normalized_sentence = helpers.normalize_string(sentence)
        # normalized_sentence = [normalized_sentence]
        (
            beam_translation,
            greedy_translation,
            sen_loss,
            beam_loss,
            ref_pad_cnt,
            valid_token,
        ) = evaluate_sentence(sentence, ref_sentence, max_len=max_len)
        ref_pad_cnt_total += ref_pad_cnt

        loss += sen_loss
        total_token += valid_token
        # beam_loss_total += beam_loss
        # print(f"Average loss: {loss/(idx+1):.4f}")
        # Use beam translation as default output
        # print(beam_translation)

        # results.append(
        #     {"source": sentence, "beam": beam_translation, "greedy": greedy_translation}
        # )
        results.append({"source": sentence, "greedy": greedy_translation})
        if idx % 100 == 0:  # Print samples every 100 sentences
            print(f"{idx}/{len(sentences)} | latest avg loss: {sen_loss:.4f}")
            # print(f"src: {sentence}")
            # print(f"ref: {ref_sentences[idx]}")
            # print(f"beam: {beam_translation}")
            # print(f"greedy: {greedy_translation}")
            # print("=" * 50)
        idx += 1
    # loss /= total_token
    print(f"Final Average Greedy Loss: {loss/len(sentences):.4f}")
    # print(f"Final Average Greedy Loss: {loss:.4f}")
    print(f"Final Greedy Perplexity: {math.exp(loss/len(sentences)):.2f}")
    # print(f"Final Greedy Perplexity: {math.exp(loss):.2f}")
    # print(f"Final Average Beam Loss: {beam_loss_total/len(sentences):.4f}")
    # print(f"Final Beam Perplexity: {math.exp(beam_loss_total/len(sentences)):.2f}")

    # Save results to output file if specified
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"{result['greedy']}\n")

        # Save greedy results to separate file
        # greedy_output_path = output_file_path.replace(".txt", "_greedy.txt")
        # with open(greedy_output_path, "w", encoding="utf-8") as f:
        #     for result in results:
        #         f.write(f"{result['greedy']}\n")

        print(f"Greedy search results saved to: {output_file_path}")
        # print(f"Greedy search results saved to: {greedy_output_path}")

    print("Total PAD in reference: ", ref_pad_cnt_total)
    return results


def greedy_decode(
    decoder_context,
    decoder_hidden,
    encoder_outputs,
    max_len,
    targets,
    is_teaching_force=False,
):
    # Run through decoder
    decoded_words = []
    encoder_len = encoder_outputs.size(0)
    decoder_attentions = torch.zeros(max_len, encoder_len)
    decoder_input = (
        torch.LongTensor(1, 1).fill_(Language.eos_token).to(device)
    )  # Use SOS token
    loss = 0
    valid_token = 0
    for di in range(max_len):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
        mask = targets[di] != Language.pad_token
        decoder_output = decoder_output[mask]
        if decoder_output.size(0) == 0:
            break
        # decoder_output = decoder_output.squeeze(0)  # [1, batch, vocab] -> [batch, vocab]
        # _loss = F.nll_loss(
        #     decoder_output,
        #     targets[di][mask],
        #     ignore_index=Language.pad_token,
        #     reduction="sum",
        # ).item()
        # _loss = F.nll_loss(decoder_output, targets[di]).item()
        _loss = F.nll_loss(
            decoder_output, targets[di], ignore_index=Language.pad_token
        ).item()

        loss += _loss / decoder_output.size(0) if not math.isnan(_loss) else 0  # or nan
        # loss += _loss
        valid_token += 1 if not math.isnan(_loss) else 0
        # valid_token += mask.sum().item()

        # decoder_attentions[di, : decoder_attention.size(2)] += (
        #     decoder_attention.squeeze(0).squeeze(0).cpu().data
        # )
        # Choose top word from output
        topv, topi = decoder_output.data.topk(1, dim=1)
        ni = topi.item()
        if ni == Language.eos_token:
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = topi
        if is_teaching_force:
            decoder_input = targets[di].view(1, -1)
    loss /= valid_token
    return (
        decoded_words,
        decoder_attentions[: di + 1, : encoder_outputs.size(0)],
        loss,
        valid_token,
    )


def beam_decode(decoder_context, decoder_hidden, encoder_outputs, max_len, beam_size=5):
    batch_size = 1  # Single sentence evaluation
    vocab_size = output_lang.n_words
    # [1, batch_size x beam_size]
    decoder_input = (
        torch.ones(batch_size * beam_size, dtype=torch.long, device=device)
        * Language.eos_token  # Use SOS token
    )

    # [num_layers, batch_size x beam_size, hidden_size]
    decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
    decoder_context = decoder_context.repeat(1, beam_size, 1)

    encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

    # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
    batch_position = (
        torch.arange(0, batch_size, dtype=torch.long, device=device) * beam_size
    )

    score = torch.ones(batch_size * beam_size, device=device) * -float("inf")
    score.index_fill_(
        0, torch.arange(0, batch_size, dtype=torch.long, device=device) * beam_size, 0.0
    )

    # Initialize Beam that stores decisions for backtracking
    beam = Beam(batch_size, beam_size, max_len, batch_position, Language.eos_token)

    for i in range(max_len):
        decoder_output, decoder_context, decoder_hidden, _ = decoder(
            decoder_input, decoder_context, decoder_hidden, encoder_outputs
        )
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
            score.data.masked_fill_(eos_idx, -float("inf"))

    prediction, final_score, length = beam.backtrack()
    return prediction, final_score, length


def assemble_sentence(words):
    final_words = list()
    for word in words:
        if word in ["<SOS>", "<PAD>"]:
            continue
        elif word == "<EOS>":
            break
        final_words.append(word)
    sentence = " ".join(final_words)
    return sentence


def print_sentence(words, lengths=None, mode="greedy"):
    if mode == "greedy":
        print("greedy > %s" % assemble_sentence(words))
    elif mode == "beam":
        for i, (length, ids) in enumerate(zip(lengths, words.tolist())):
            cur_words = [output_lang.index2word[id] for id in ids[:length]]
            sentence = assemble_sentence(cur_words)
            print("beam %d > %s" % (i, sentence))


# Evaluate sentences from input file
if args.input_file:
    evaluate_file(args.input_file, args.input_ref_file, args.output_file, args.max_len)
else:
    print("Please provide --input_file argument")
    exit(1)
