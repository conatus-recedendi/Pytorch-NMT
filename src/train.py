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
from torch.cuda.amp import autocast, GradScaler

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


def train(
    input, target, encoder, decoder, encoder_opt, decoder_opt, criterion, scaler=None
):
    # Initialize optimizers and loss
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0
    # input is listattribute 'size'
    batch_size = input.size(0)

    # Get input and target seq lengths
    target_length = target.size(1)

    # Run through encoder
    encoder_hidden = encoder.init_hidden(device, batch_size)  # Pass actual batch size
    input = input.squeeze(-1)  # [batch_size, seq_len] - remove the last dimension
    # print(input.shape, encoder_hidden.shape)
    encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)

    # Prepare input and output variables
    # decoder_input = torch.LongTensor([0]).to(device)
    decoder_input = torch.LongTensor(batch_size, 1).fill_(0).to(device)
    #
    decoder_context = torch.zeros(1, batch_size, decoder.hidden_size).to(device)

    decoder_hidden = encoder_hidden

    # Scheduled sampling
    use_teacher_forcing = random.random() < args.teacher_forcing_ratio

    # Pre-allocate tensors for better performance
    all_decoder_outputs = []

    if use_teacher_forcing:
        # Feed target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
            all_decoder_outputs.append(decoder_output)
            decoder_input = target[:, di].unsqueeze(1)  # [batch_size, 1]

        # Compute loss for all timesteps at once
        decoder_outputs_tensor = torch.stack(
            all_decoder_outputs, dim=1
        )  # [batch_size, seq_len, vocab_size]
        decoder_outputs_flat = decoder_outputs_tensor.view(
            -1, decoder_outputs_tensor.size(-1)
        )
        target_flat = target.view(-1)

        # NLLLoss with ignore_index will handle padding automatically
        loss = criterion(decoder_outputs_flat, target_flat)
    else:
        # Use previous prediction as next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
            target_di = target[:, di].squeeze()
            loss += criterion(decoder_output, target_di)

            topv, topi = decoder_output.data.topk(1, dim=1)
            decoder_input = topi  # [batch_size, 1]

            # Check if all sequences have reached EOS token (simplified)
            if (topi == Language.eos_token).all():
                break

    # Backpropagation
    if scaler is not None:
        scaler.scale(loss).backward()
        # Apply gradient clipping before stepping
        scaler.unscale_(encoder_opt)
        scaler.unscale_(decoder_opt)
        nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        scaler.step(encoder_opt)
        scaler.step(decoder_opt)
        scaler.update()
    else:
        loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        decoder_grad_norm = nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        encoder_opt.step()
        decoder_opt.step()

    # Check for NaN
    if torch.isnan(loss):
        print(f"NaN detected! Target length: {target_length}, Loss: {loss.item()}")
        print(f"Target shape: {target.shape}, Input shape: {input.shape}")
        return float("inf")

    return loss.item() / target_length


input_lang, output_lang, pairs = etl.prepare_data(args.language)

print(input_lang)
# Initialize models
encoder = EncoderRNN(
    128,  # max batch size for init_hidden
    input_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.n_layers,
    args.dropout,
)

decoder = AttentionDecoderRNN(
    128,  # max batch size for initialization
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
encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
criterion = nn.NLLLoss(ignore_index=0)  # Ignore padding tokens

# Initialize mixed precision scaler
scaler = GradScaler() if device.type == "cuda" else None


# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# Begin training
lr = args.lr
progress = 0.0
avg_loss = 0.0
for epoch in range(1, args.n_epochs + 1):
    # Get training data for this cycle
    if epoch > 5:
        lr = args.lr / (2 ** (epoch - 5))  # More efficient learning rate decay
        for param_group in encoder_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in decoder_optimizer.param_groups:
            param_group["lr"] = lr

    batch_size = 128  # Restore larger batch size for efficiency
    # print("hi\n")
    epoch_loss = 0.0
    batch_count = 0

    for _ in range(len(pairs) // batch_size):
        # Print progress every 100 batches to reduce I/O overhead
        if _ % 1 == 0:
            progress = (
                ((_ + 1)) / ((len(pairs) // batch_size) * args.n_epochs)
                + (epoch - 1) / args.n_epochs
            ) * 100
            expected_time_sec = (
                (time.time() - start)
                / (_ + 1)
                * ((len(pairs) // batch_size) * args.n_epochs - (_ + 1) * epoch)
            )
            expected_time_str = helpers.format_time(expected_time_sec)
            print(
                "%cEpoch: %d/%d, Loss: %f, Progress: %f%%, Expected Time: %s"
                % (
                    13,
                    epoch,
                    args.n_epochs,
                    avg_loss if batch_count > 0 else 0,
                    progress,
                    expected_time_str,
                ),
                end="\r",
            )
            sys.stdout.flush()
        pair_batch = pairs[_ * batch_size : (_ + 1) * batch_size]
        training_pair_batch = etl.tensor_from_pair_batch(
            pair_batch, input_lang, output_lang, device
        )
        input = training_pair_batch[0]
        target = training_pair_batch[1]
        input = pad_sequence(input, batch_first=True)
        target = pad_sequence(target, batch_first=True)

        # Run the train step
        if scaler is not None:
            with autocast():
                batch_loss = train(
                    input,
                    target,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                    scaler,
                )
        else:
            batch_loss = train(
                input,
                target,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
                None,
            )

        epoch_loss += batch_loss
        batch_count += 1
        # avg_loss = epoch_loss / batch_count
        avg_loss = batch_loss

        # Check for problematic loss values
        if torch.isnan(torch.tensor(batch_loss)):
            print(f"Problematic loss detected: {batch_loss}")
            print(f"Learning rate: {encoder_optimizer.param_groups[0]['lr']}")
            break
    # print(input.shape)

    # Keep track of loss
    print_loss_total += avg_loss
    plot_loss_total += avg_loss

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
