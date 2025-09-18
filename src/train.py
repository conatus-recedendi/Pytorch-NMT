import sys
import argparse
import etl
import helpers
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from attention_decoder import AttentionDecoderRNN
from encoder import EncoderRNN
from language import Language
from torch.nn.utils.rnn import pad_sequence

# Parse argument for language to train
parser = argparse.ArgumentParser()
parser.add_argument(
    "--attn_model", type=str, help="attention type: dot, general, concat"
)
parser.add_argument("--embedding_size", type=int)
parser.add_argument("--hidden_size", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--dropout", type=float)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
parser.add_argument("--clip", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--n_epochs", type=int)
parser.add_argument("--plot_every", type=int)
parser.add_argument("--print_every", type=int)
parser.add_argument("--language", type=str, help="specific which language.")
parser.add_argument("--input", type=str, help="src -> tgt")
parser.add_argument("--device", type=str, help="cpu or cuda")
parser.add_argument("--seed", type=str, help="random seed")
parser.add_argument("--batch_size", type=int, help="batch size")
args = parser.parse_args()

print(sys.argv)

torch.random.manual_seed(args.seed)
device = torch.device(args.device)
print("device: ", device)

# helpers.validate_language(args.language)


def train(input, target, encoder, decoder, encoder_opt, decoder_opt, criterion):
    # Initialize optimizers and loss
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0
    # input is listattribute 'size'
    batch_size = input.size(0)

    # Get input and target seq lengths
    target_length = target.size(1)

    # Run through encoder
    encoder_hidden = encoder.init_hidden(device)
    print(input.shape, encoder_hidden.shape)
    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)

    # Prepare input and output variables
    # decoder_input = torch.LongTensor([0]).to(device)
    decoder_input = torch.LongTensor([0] * batch_size).to(device)
    #
    decoder_context = torch.zeros(batch_size, 1, decoder.hidden_size).to(device)
    decoder_hidden = encoder_hidden

    # Scheduled sampling
    use_teacher_forcing = random.random() < args.teacher_forcing_ratio
    if use_teacher_forcing:
        # Feed target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
            loss += criterion(decoder_output, target[di])
            decoder_input = target[di]
    else:
        # Use previous prediction as next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
            # decoder_output: [batch_size, 1, tgt_vocab_size]
            loss += criterion(decoder_output, target[di])

            topv, topi = decoder_output.data.topk(1, dim=1)

            decoder_input = topi

            if topi.item() == Language.eos_token:
                break

    # Backpropagation
    loss.backward()
    #  print(list(encoder.parameters()))
    #  print(args.clip)
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
    encoder_opt.step()
    decoder_opt.step()

    return loss.item() / target_length


input_lang, output_lang, pairs = etl.prepare_data(args.language)

print(input_lang)
# Initialize models
encoder = EncoderRNN(
    args.batch_size,
    input_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.n_layers,
    args.dropout,
)

decoder = AttentionDecoderRNN(
    args.batch_size,
    output_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.attn_model,
    args.n_layers,
    args.dropout,
)
# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Initialize optimizers and criterion
criterion = nn.NLLLoss()


# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# Begin training
lr = args.lr
progress = 0.0
for epoch in range(1, args.n_epochs + 1):
    # Get training data for this cycle
    if epoch > 5:
        lr = lr / 2
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    batch_size = 128
    print("hi\n")

    for _ in range(len(pairs) // batch_size):
        progress = (_ + 1) / ((len(pairs) // batch_size) * epoch) * 100

        print(
            "%cEpoch: %d/%d, Progress: %f%%" % (13, epoch, args.n_epochs, progress),
            end="\r",
        )
        sys.stdout.flush()
        pair_batch = pairs[_ * batch_size : (_ + 1) * batch_size]
        training_pair_batch = etl.tensor_from_pair(
            pair_batch, input_lang, output_lang, device
        )
        input = training_pair_batch[0]
        target = training_pair_batch[1]
        # print(input)
        # input is list to -> tensor
        # input = torch.cat(input, dim=1)
        # target = torch.cat(target, dim=1)
        input = pad_sequence(input, batch_first=True)
        target = pad_sequence(target, batch_first=True)
        # input = torch.stack(input, dim=0)
        # target = torch.stack(target, dim=0)
        print(input.shape)
        # Run the train step
        loss = train(
            input,
            target,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )

    print(input.shape)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0:
        continue

    if epoch % args.print_every == 0:
        print_loss_avg = print_loss_total / args.print_every
        print_loss_total = 0
        time_since = helpers.time_since(start, epoch / args.n_epochs)
        print(
            "%s (%d %d%%) %.4f"
            % (time_since, epoch, epoch / args.n_epochs * 100, print_loss_avg)
        )

    if epoch % args.plot_every == 0:
        plot_loss_avg = plot_loss_total / args.plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0


# Save our models
torch.save(encoder.state_dict(), "./data/encoder_params_{}".format(args.language))
torch.save(decoder.state_dict(), "./data/decoder_params_{}".format(args.language))
torch.save(
    decoder.attention.state_dict(), "./data/attention_params_{}".format(args.language)
)

# Plot loss
helpers.show_plot(plot_losses)
