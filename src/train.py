import sys
import argparse
import etl
import helpers
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from attention_decoder import AttentionDecoderRNN
from encoder import EncoderRNN
from language import Language
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

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
parser.add_argument("--local", type=str, help="local-m, local-p, None", default=None)
args = parser.parse_args()

print(sys.argv)

torch.random.manual_seed(args.seed)
device = torch.device(args.device)
print("device: ", device)

# helpers.validate_language(args.language)


# Perplexity calculation functions
def load_test_data(
    en_file, de_file, input_lang, output_lang, device="cpu", max_samples=1000
):
    """테스트 데이터를 로드하고 tensor로 변환"""
    test_pairs = []

    with open(en_file, "r", encoding="utf-8") as f_en, open(
        de_file, "r", encoding="utf-8"
    ) as f_de:

        for i, (en_line, de_line) in enumerate(zip(f_en, f_de)):
            if i >= max_samples:  # 계산 시간을 위해 샘플 수 제한
                break
            en_line = en_line.strip()
            de_line = de_line.strip()
            if en_line and de_line:
                test_pairs.append((en_line, de_line))

    # 텐서로 변환
    test_inputs = []
    test_targets = []

    for en_sent, de_sent in test_pairs:
        # 문장을 인덱스로 변환
        en_indexes = [
            input_lang.word2index.get(word, input_lang.word2index.get("<UNK>", 3))
            for word in en_sent.split()
        ]
        de_indexes = [
            output_lang.word2index.get(word, output_lang.word2index.get("<UNK>", 3))
            for word in de_sent.split()
        ]

        en_indexes.append(Language.eos_token)  # EOS token
        de_indexes.append(Language.eos_token)  # EOS token

        en_tensor = torch.LongTensor(en_indexes).view(-1, 1).to(device)
        de_tensor = torch.LongTensor(de_indexes).view(-1, 1).to(device)

        test_inputs.append(en_tensor)
        test_targets.append(de_tensor)

    return test_inputs, test_targets


def calculate_perplexity(
    encoder, decoder, test_inputs, test_targets, criterion, device
):
    """테스트 데이터에 대한 Perplexity 계산"""
    # 현재 training 상태 저장
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training

    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        # 배치 처리를 위해 패딩
        batch_size = min(32, len(test_inputs))  # 메모리 고려해서 작은 배치 사용
        # batch_size = 1

        for i in range(0, len(test_inputs), batch_size):
            batch_inputs = test_inputs[i : i + batch_size]
            batch_targets = test_targets[i : i + batch_size]

            # 패딩 적용
            input_batch = pad_sequence(
                batch_inputs, batch_first=True, padding_value=2
            )  # PAD token
            target_batch = pad_sequence(
                batch_targets, batch_first=True, padding_value=2
            )  # PAD token

            actual_batch_size = input_batch.size(0)
            target_length = target_batch.size(1)

            # Forward pass
            encoder_hidden = encoder.init_hidden(device, actual_batch_size)
            input_batch = input_batch.squeeze(-1)
            encoder_outputs, encoder_hidden = encoder(input_batch, encoder_hidden)

            # Decoder
            # <SOS> 토큰으로 초기화
            decoder_input = (
                torch.LongTensor(actual_batch_size, 1)
                .fill_(Language.sos_token)
                .to(device)
            )
            decoder_context = torch.zeros(1, actual_batch_size, decoder.hidden_size).to(
                device
            )
            decoder_hidden = encoder_hidden

            batch_loss = 0
            valid_tokens = 0

            # No teacher forcing for evaluation - use model's own predictions
            all_decoder_outputs = []
            for di in range(target_length):
                # print(decoder_input.shape, decoder_context.shape, decoder_hidden.shape)
                decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                    decoder(
                        decoder_input, decoder_context, decoder_hidden, encoder_outputs
                    )
                )

                # target_di = target_batch[:, di].squeeze()
                all_decoder_outputs.append(decoder_output)
                decoder_input = target_batch[:, di].unsqueeze(1)  # [batch_size, 1]
                # 패딩 토큰(2) 제외 - 일반적인 배치 크기 처리
            decoder_outputs_tensor = torch.stack(
                all_decoder_outputs, dim=1
            )  # [batch_size, seq_len, vocab_size]
            decoder_outputs_flat = decoder_outputs_tensor.view(
                -1, decoder_outputs_tensor.size(-1)
            )
            target_flat = target_batch.view(-1)
            # target_di shape: [batch_size] when squeezed
            loss = criterion(decoder_outputs_flat, target_flat)
            valid_tokens = (target_flat != 2).sum().item()  # PAD token = 2
            # loss = loss / valid_tokens if valid_tokens > 0 else loss

            total_loss += loss
            total_tokens += valid_tokens

    # 원래 training 상태로 복원
    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    # avg_loss /= len(test_inputs)
    print(f"Avg Loss: {avg_loss:.4f}, Total Tokens: {total_tokens}")
    print(f"Total Loss: {total_loss:.4f}")

    # Debug perplexity calculation
    if avg_loss > 10:
        print(
            f"WARNING: Average loss is very high ({avg_loss:.4f}), capping at 10 for perplexity"
        )
        perplexity = math.exp(10)  # Cap to prevent overflow
    else:
        perplexity = math.exp(avg_loss)

    print(
        f"Raw perplexity calculation: exp({min(avg_loss, 10):.4f}) = {perplexity:.4f}"
    )
    return perplexity


def train(
    input, target, encoder, decoder, encoder_opt, decoder_opt, criterion, scaler=None
):
    # Initialize optimizers and loss
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = torch.tensor(0.0, device=device, requires_grad=True)  # Initialize properly
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
    decoder_input = torch.LongTensor(batch_size, 1).fill_(Language.sos_token).to(device)
    #
    decoder_context = torch.zeros(1, batch_size, decoder.hidden_size).to(device)

    decoder_hidden = encoder_hidden

    # Scheduled sampling - enable teacher forcing for better learning
    use_teacher_forcing = random.random() < args.teacher_forcing_ratio

    # Pre-allocate tensors for better performance

    if use_teacher_forcing:
        # Teacher forcing: Feed target as the next input
        all_decoder_outputs = []
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
        valid_tokens = (target_flat != 2).sum().item()  # PAD token = 2
        loss = loss / valid_tokens if valid_tokens > 0 else loss
    else:
        # No teacher forcing: use previous prediction as next input
        loss_sum = 0
        valid_tokens = 0
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = (
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            )
            target_di = target[:, di].squeeze()
            non_pad_mask = target_di != 2  # PAD token = 2

            # Check for NaN in intermediate values
            if torch.isnan(decoder_output).any():
                print(f"NaN in decoder_output at step {di}")
                return float("inf")

            # Decoder output is already log_softmax applied
            if non_pad_mask.any():
                step_loss = criterion(
                    decoder_output[non_pad_mask], target_di[non_pad_mask]
                )
                loss_sum += step_loss
                valid_tokens += non_pad_mask.sum().item()

            if torch.isnan(loss_sum):
                print(f"NaN in loss_sum at step {di}")
                return float("inf")

            topv, topi = decoder_output.data.topk(1, dim=1)
            decoder_input = topi  # [batch_size, 1]

        loss = loss_sum / valid_tokens if valid_tokens > 0 else loss_sum

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

    return loss.item()  # Don't divide by target_length again


input_lang, output_lang, pairs = etl.prepare_data(args.language)

print(input_lang)
# Initialize models
encoder = EncoderRNN(
    args.batch_size,  # max batch size for init_hidden
    input_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.n_layers,
    args.dropout,
)

decoder = AttentionDecoderRNN(
    args.batch_size,  # max batch size for initialization
    output_lang.n_words,
    args.embedding_size,
    args.hidden_size,
    args.attn_model,
    args.n_layers,
    args.dropout,
    args.local,
)
# Move models to device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Initialize optimizers and criterion
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)
criterion = nn.NLLLoss(ignore_index=2, reduction="sum")  # Ignore padding tokens (PAD=2)

# Initialize mixed precision scaler - Disable for debugging
scaler = None  # GradScaler() if device.type == "cuda" else None

# Load test data for perplexity calculation
test_inputs = None
test_targets = None
try:
    test_inputs, test_targets = load_test_data(
        "./rewrite/test.14.en",
        "./rewrite/test.14.de",
        input_lang,
        output_lang,
        device,
        max_samples=2716,
    )
    print(f"Loaded {len(test_inputs)} test samples for perplexity calculation")
except FileNotFoundError:
    print("Test files not found. Perplexity calculation will be skipped.")

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# Begin training
lr = args.lr
progress = 0.0
avg_loss = 0.0
total_batch_count = 0
batch_size = args.batch_size
print(
    "max total_Batch_count: ",
    (len(pairs) // batch_size) * args.n_epochs,
    len(pairs),
    batch_size,
    args.n_epochs,
)
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
        total_batch_count += 1
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
                "%cEpoch: %d/%d, Batch: %d, Loss: %f, Progress: %f%%, Expected Time: %s"
                % (
                    13,
                    epoch,
                    args.n_epochs,
                    total_batch_count,
                    avg_loss if batch_count > 0 else 0,
                    progress,
                    expected_time_str,
                ),
                end="\r",
                file=sys.stderr,
            )
            sys.stdout.flush()
        pair_batch = pairs[_ * batch_size : (_ + 1) * batch_size]
        training_pair_batch = etl.tensor_from_pair_batch(
            pair_batch, input_lang, output_lang, device
        )
        input = training_pair_batch[0]
        target = training_pair_batch[1]
        input = pad_sequence(input, batch_first=True, padding_value=2)  # PAD token
        target = pad_sequence(target, batch_first=True, padding_value=2)  # PAD token

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

        # Calculate perplexity every 100 batches
        if total_batch_count % 1000 == 0 and test_inputs is not None:
            print(f"\n\nCalculating perplexity at batch {total_batch_count}...")
            # Simple perplexity calculation using current loss
            # Perplexity = exp(average_loss)
            current_ppl = calculate_perplexity(
                encoder, decoder, test_inputs, test_targets, criterion, device
            )
            # Ensure models are back in training mode
            encoder.train()
            decoder.train()
            # current_ppl = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
            print(
                f"Approximate Perplexity at batch {total_batch_count}: {current_ppl:.4f}"
            )
            print("Continuing training...\n")

        # Check for problematic loss values
        if torch.isnan(torch.tensor(batch_loss)):
            print(f"Problematic loss detected: {batch_loss}")
            print(f"Learning rate: {encoder_optimizer.param_groups[0]['lr']}")
            break
    # print(input.shape)

    # Keep track of loss
    print_loss_total += avg_loss
    plot_loss_total += avg_loss
    # if total_batch_count % 10000 == 0:
    #     # get test perplexity for Figure 5

    if epoch == 0:
        continue

    if epoch % args.print_every == 0:
        print_loss_avg = print_loss_total / args.print_every
        print_loss_total = 0
        time_since = helpers.time_since(start, epoch / args.n_epochs)
        print(
            "%s (%d %d%%) %.4f"
            % (time_since, epoch, epoch / args.n_epochs * 100, print_loss_avg),
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
