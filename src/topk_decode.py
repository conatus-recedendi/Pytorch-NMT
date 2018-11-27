#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tok Decode, fork
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
"""
import torch
import torch.nn.functional as F

class TopKDecode(torch.nn.Module):
    r"""
    Top-beam_size decoding with beam search.
    Args:
        decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
        beam_size (int): Size of the beam.
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features
          in the decoder_hidden state `h` of encoder. Used as the initial decoder_hidden state of the decoder.
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from decoder decoder_hidden state
          (default is `torch.nn.functional.log_softmax`).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_len, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last decoder_hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
          representing lengths of output sequences, *topk_length*: list of integers representing lengths of beam search
          sequences, *sequence* : list of sequences, where each sequence is a list of predicted token IDs,
          *topk_sequence* : list of beam search sequences, each beam is a list of token IDs, *inputs* : target
          outputs if provided for decoding}.
    """

    def __init__(self,
                 decoder,
                 hidden_size,
                 beam_size,
                 vocab_size,
                 sos_id,
                 eos_id,
                 device):
        super(TopKDecode, self).__init__()
        self.decoder = decoder
        self.beam_size = beam_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.device = device

    def forward(self,
                decoder_context=None,
                decoder_hidden=None,
                encoder_outputs=None,
                max_len=10,
                batch_size=1):

        # [batch_size * beam_size, 1]
        self.pos_index = (torch.LongTensor(range(batch_size)) * self.beam_size).view(-1, 1).to(device)

        # Inflate the initial decoder_hidden states to be of size: batch_size*beam_size x h

        # ... same idea for encoder_outputs and decoder_outputs
        encoder_outputs = encoder_outputs.repeat(1, self.beam_size, 1) # [max_len, batch_size * beam_size, hidden_size]
        decoder_hidden = decoder_hidden.repeat(1, self.beam_size, 1)
        decoder_context = decoder_context.repeat(1, self.beam_size, 1) # [num_layer, batch_size * beam_size, hidden_size]

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top beam_size
        sequence_scores = torch.Tensor(batch_size * self.beam_size, 1).to(device)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.beam_size for i in range(0, batch_size)]), 0.0)

        # Initialize the decoder_input vector
        decoder_input = torch.LongTensor([[self.sos_id] * batch_size * self.beam_size]).to(device) # [1, beam_size * batch_size]

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _ in range(0, max_len):
            # output: [batch_size * beam_size, vocab_size]
            #  print(decoder_input.shape)
            #  print(decoder_context.shape)
            #  print(decoder_hidden.shape)
            #  print(encoder_outputs.shape)
            output, decoder_context, decoder_hidden, _ = self.decoder(decoder_input,
                                                                decoder_context,
                                                                decoder_hidden,
                                                                encoder_outputs)

            stored_outputs.append(output)

            # To get the full sequence scores for the new candidates, add the local
            # scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = sequence_scores.repeat(1, self.vocab_size)
            sequence_scores += output # [batch_size * beam_size, vocab_size]
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.beam_size, dim=1)
            # [batch_size, beam_size]

            # Reshape decoder_input = (bk, 1) and sequence_scores = (bk, 1)
            decoder_input = (candidates % self.vocab_size).view(batch_size * self.beam_size, 1) # [beam_size * batch_size, 1]
            sequence_scores = scores.view(batch_size * self.beam_size, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.vocab_size + self.pos_index.expand_as(candidates)).view(batch_size * self.beam_size, 1)
            decoder_hidden = decoder_hidden.index_select(1, predecessors.squeeze())
            decoder_context = decoder_context.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = decoder_input.data.eq(self.eos_id)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(decoder_input)
            stored_hidden.append(decoder_hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, score, topk_length, topk_sequence = self._backtrack(stored_outputs,
                                                    stored_hidden,
                                                    stored_predecessors,
                                                    stored_emitted_symbols,
                                                    stored_scores,
                                                    batch_size,
                                                    max_len)

        #  print(output)
        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        decoder_hidden = h_n[:, :, 0, :]
        #  print(h_t)
        #  print(topk_length)
        #  for item in topk_sequence:
            #  print(item.tolist())
        #  print(topk_sequence)

        metadata = {}
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = score
        metadata['topk_length'] = topk_length
        metadata['topk_sequence'] = topk_sequence
        metadata['length'] = [seq_len[0] for seq_len in topk_length]
        metadata['sequence'] = [seq[0] for seq in topk_sequence]
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, batch_size, max_len):
        """Backtracks over batch to generate optimal beam_size-sequences.
        Args:
            nw_output [(batch*beam_size, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*beam_size, hidden_size)] * sequence_length: A Tensor of decoder_hidden states from network
            predecessors [(batch*beam_size)] * sequence_length: A Tensor of predecessors
            symbols [(batch*beam_size)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*beam_size)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            batch_size: Size of the batch
            hidden_size: Size of the decoder_hidden state
        Returns:
            output [(batch, beam_size, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the decoder, for every n = [0, ... , seq_len - 1]
            h_t [(batch, beam_size, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the decoder, for every n = [0, ... , seq_len - 1]
            h_n(batch, beam_size, hidden_size): A Tensor containing the last decoder_hidden state for all top-beam_size sequences.
            score [batch, beam_size]: A list containing the final scores for all top-beam_size sequences
            length [batch, beam_size]: A list specifying the length of each sequence in the top-beam_size candidates
            topk_sequence (batch, beam_size, sequence_len): A Tensor containing predicted sequence
        """

        output = list()
        h_t = list()
        topk_sequence = list()
        # Placeholder for last decoder_hidden state of top-beam_size sequences.
        # If a (top-beam_size) sequence ends early in decoding, `h_n` contains
        # its decoder_hidden state when it sees eos_id.  Otherwise, `h_n` contains
        # the last decoder_hidden state of decoding.
        h_n = torch.zeros(nw_hidden[0].size())
        topk_length = [[max_len] * self.beam_size for _ in range(batch_size)]  # Placeholder for lengths of top-beam_size sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(batch_size, self.beam_size).topk(self.beam_size)
        # initialize the sequence scores with the sorted last step beam scores
        score = sorted_score.clone()

        batch_eos_found = [0] * batch_size   # the number of eos_id found
                                    # in the backward loop below for each batch

        t = max_len - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with batch_size*beam_size as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(batch_size * self.beam_size)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see eos_id earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see eos_id early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an eos_id in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.eos_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the eos_id symbol for both variables
                    # with batch_size*beam_size as the first dimension, and batch_size, beam_size for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.beam_size)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.beam_size - (batch_eos_found[b_idx] % self.beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                    h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    score[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    topk_length[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            topk_sequence.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        score, re_sorted_idx = score.topk(self.beam_size)
        for b_idx in range(batch_size):
            topk_length[b_idx] = [topk_length[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(batch_size * self.beam_size)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(batch_size, self.beam_size, -1) for step in reversed(output)]
        topk_sequence = [step.index_select(0, re_sorted_idx).view(batch_size, self.beam_size, -1) for step in reversed(topk_sequence)]
        h_t = [step.index_select(1, re_sorted_idx).view(-1, batch_size, self.beam_size, self.hidden_size) for step in reversed(h_t)]
        h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, batch_size, self.beam_size, self.hidden_size)
        score = score.data

        return output, h_t, h_n, score, topk_length, topk_sequence

