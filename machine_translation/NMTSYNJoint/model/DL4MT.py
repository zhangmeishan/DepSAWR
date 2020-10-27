import torch.nn as nn
import torch.nn.functional as F
from module import Init
from module.Basic import BottleLinear as Linear
from module.Embeddings import Embeddings
from module.CGRU import CGRUCell
from module.RNN import RNN
from module.Utils import *
from module.ScaleMix import *


class Encoder(nn.Module):
    def __init__(self,
                 parser_config,
                 src_vocab,
                 input_size,
                 hidden_size
                 ):

        super(Encoder, self).__init__()

        self.src_vocab = src_vocab
        # Use PAD
        self.embeddings = Embeddings(num_embeddings=src_vocab.max_n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     padding_idx=src_vocab.PAD,
                                     add_position_embedding=False)

        self.transformer_emb = nn.Linear(parser_config.word_dims, input_size, bias=False)
        parser_dim = 2 * parser_config.lstm_hiddens
        transformer_lstm = []
        for layer in range(parser_config.lstm_layers):
            transformer_lstm.append(nn.Linear(parser_dim, input_size, bias=False))
        self.transformer_lstm = nn.ModuleList(transformer_lstm)

        parser_mlp_dim = parser_config.mlp_arc_size + parser_config.mlp_rel_size
        self.transformer_dep = nn.Linear(parser_mlp_dim, input_size, bias=False)
        self.transformer_head = nn.Linear(parser_mlp_dim, input_size, bias=False)

        self.parser_lstm_layers = parser_config.lstm_layers
        self.synscale = ScalarMix(mixture_size=3+parser_config.lstm_layers)

        self.gru = RNN(type="gru", batch_first=True, input_size=2*input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def get_embeddings(self, seqs):
        emb = self.embeddings(seqs)
        return emb

    def forward(self, xs, synxs, xlengths):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = xs.data.eq(self.src_vocab.PAD)

        emb = self.get_embeddings(xs)

        syn_idx = 0
        x_syns = []
        x_syn_emb = self.transformer_emb(synxs[syn_idx])
        x_syns.append(x_syn_emb)
        syn_idx += 1

        for layer in range(self.parser_lstm_layers):
            x_syn_lstm = self.transformer_lstm[layer].forward(synxs[syn_idx])
            syn_idx += 1
            x_syns.append(x_syn_lstm)

        x_syn_dep = self.transformer_dep(synxs[syn_idx])
        x_syns.append(x_syn_dep)
        syn_idx += 1

        x_syn_head = self.transformer_head(synxs[syn_idx])
        x_syns.append(x_syn_head)
        syn_idx += 1

        syn_hidden = self.synscale(x_syns)

        encoder_input = torch.cat([emb, syn_hidden], 2)

        ctx, _ = self.gru(encoder_input, xlengths)

        return ctx, x_mask


class Decoder(nn.Module):

    def __init__(self,
                 tgt_vocab,
                 input_size,
                 hidden_size,
                 bridge_type="mlp",
                 dropout_rate=0.0):

        super(Decoder, self).__init__()

        self.tgt_vocab = tgt_vocab

        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = hidden_size * 2

        self.embeddings = Embeddings(num_embeddings=tgt_vocab.max_n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     padding_idx=tgt_vocab.PAD,
                                     add_position_embedding=False)

        self.cgru_cell = CGRUCell(input_size=input_size, hidden_size=hidden_size)

        self.linear_input = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.linear_ctx = nn.Linear(in_features=hidden_size * 2, out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

    def _reset_parameters(self):
        Init.default_init(self.linear_input.weight)
        Init.default_init(self.linear_hidden.weight)
        Init.default_init(self.linear_ctx.weight)

    def _build_bridge(self):
        if self.bridge_type == "mlp":
            self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
            Init.default_init(self.linear_bridge.weight.data)
        elif self.bridge_type == "zero":
            pass
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

    def get_embeddings(self, seqs):
        emb = self.embeddings(seqs)

        return emb

    def init_decoder(self, context, mask):

        # Generate init hidden
        if self.bridge_type == "mlp":
            no_pad_mask = 1.0 - mask.float()
            ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
            dec_init = torch.tanh(self.linear_bridge(ctx_mean))
        elif self.bridge_type == "zero":
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

        dec_cache = self.cgru_cell.compute_cache(context)

        return dec_init, dec_cache

    def forward(self, y, context, context_mask, hidden, one_step=False, cache=None):

        emb = self.get_embeddings(y)  # [seq_len, batch_size, dim]

        if one_step:
            (out, attn), hidden = self.cgru_cell(emb, hidden, context, context_mask, cache)
        else:
            # emb: [seq_len, batch_size, dim]
            out = []
            attn = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=0):
                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(0), hidden, context, context_mask, cache)
                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out)
            attn = torch.stack(attn)

        logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn)

        logits = torch.tanh(logits)

        logits = self.dropout(logits)  # [seq_len, batch_size, dim]

        return logits, hidden


class Generator(nn.Module):

    def __init__(self, vocab, hidden_size, shared_weight=None):
        super(Generator, self).__init__()

        self.pad = vocab.pad()
        self.n_words = vocab.max_n_words

        self.hidden_size = hidden_size

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.pad == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.pad] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """
        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class DL4MT(nn.Module):
    def __init__(self, config, p_config, src_vocab, tgt_vocab, use_gpu=True):

        super().__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_pad = src_vocab.PAD
        self.src_bos = src_vocab.BOS
        self.src_eos = src_vocab.EOS
        self.tgt_pad = tgt_vocab.PAD
        self.tgt_bos = tgt_vocab.BOS
        self.tgt_eos = tgt_vocab.EOS

        self.config = config

        self.encoder = Encoder(p_config, src_vocab,
                               input_size=config.embed_size, hidden_size=config.hidden_size)

        self.decoder = Decoder(tgt_vocab, input_size=config.embed_size, hidden_size=config.hidden_size,
                               dropout_rate=config.dropout_hidden, bridge_type=config.bridge_type)

        if config.proj_share_weight:
            generator = Generator(vocab=self.tgt_vocab,
                                  hidden_size=config.embed_size,
                                  shared_weight=self.decoder.embeddings.embeddings.weight)
        else:
            generator = Generator(vocab=self.tgt_vocab, hidden_size=config.embed_size)

        self.generator = generator
        self.use_gpu = use_gpu

        for p in self.parameters():
            nn.init.uniform_(p.data, -config.param_init, config.param_init)

    def force_teaching(self, x, src_synxs, y, lengths):

        ctx, ctx_mask = self.encoder(x, src_synxs, lengths)

        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)

        logits, _ = self.decoder(y,
                                 context=ctx,
                                 context_mask=ctx_mask,
                                 one_step=False,
                                 hidden=dec_init,
                                 cache=dec_cache)  # [tgt_len, batch_size, dim]

        return self.generator(logits.transpose(1, 0).contiguous())  # Convert to batch-first mode.

    def batch_beam_search(self, x, synxs, lengths, beam_size=5, max_steps=150):
        batch_size = x.size(0)

        ctx, ctx_mask = self.encoder(x, synxs, lengths)
        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)

        ctx = tile_batch(ctx, multiplier=beam_size, batch_dim=0)
        dec_cache = tile_batch(dec_cache, multiplier=beam_size, batch_dim=0)
        hiddens = tile_batch(dec_init, multiplier=beam_size, batch_dim=0)
        ctx_mask = tile_batch(ctx_mask, multiplier=beam_size, batch_dim=0)

        beam_mask = ctx.new(batch_size, beam_size).fill_(1.0)  # Mask of beams
        dec_memory_len = ctx.new(batch_size, beam_size).fill_(0.0)
        beam_scores = ctx.new(batch_size, beam_size).fill_(0.0)
        final_word_indices = x.new(batch_size, beam_size, 1).fill_(self.tgt_bos)

        for t in range(max_steps):
            logits, hiddens = self.decoder(y=final_word_indices.contiguous().view(batch_size * beam_size, -1)[:, -1],
                                           hidden=hiddens.view(batch_size * beam_size, -1),
                                           context=ctx,
                                           context_mask=ctx_mask,
                                           one_step=True,
                                           cache=dec_cache
                                           )

            hiddens = hiddens.view(batch_size, beam_size, -1)

            next_scores = -self.generator(logits, log_probs=True)  # [B * Bm, N]
            next_scores = next_scores.view(batch_size, beam_size, -1)
            next_scores = mask_scores(next_scores, beam_mask=beam_mask, eos_id=self.tgt_eos)

            beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)
            if t == 0 and beam_size > 1:
                # Force to select first beam at step 0
                beam_scores[:, 1:, :] = float('inf')

            # Length penalty
            normed_scores = beam_scores.detach().clone()
            # next_logits = -normed_scores
            normed_scores = normed_scores.view(batch_size, -1)

            # Get topK with beams
            # indices: [batch_size, ]
            _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
            next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
            next_word_ids = indices % vocab_size  # [batch_size, ]

            # Re-arrange by new beam indices
            beam_scores = beam_scores.view(batch_size, -1)
            beam_scores = torch.gather(beam_scores, 1, indices)

            # gather beam cache
            dec_memory_len = tensor_gather_helper(gather_indices=next_beam_ids,
                                                  gather_from=dec_memory_len,
                                                  batch_size=batch_size,
                                                  beam_size=beam_size,
                                                  gather_shape=[-1],
                                                  use_gpu=self.use_gpu)

            hiddens = tensor_gather_helper(gather_indices=next_beam_ids,
                                           gather_from=hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1],
                                           use_gpu=self.use_gpu)

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1],
                                             use_gpu=self.use_gpu)

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1],
                                                      use_gpu=self.use_gpu)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(self.tgt_eos).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                       self.tgt_pad)
            # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # update beam
            dec_memory_len += beam_mask

            final_word_indices = torch.cat((final_word_indices, torch.unsqueeze(next_word_ids, 2)), dim=2)

            if beam_mask.eq(0.0).all():
                # All the beam is finished (be zero
                break

        # Length penalty
        scores = beam_scores / dec_memory_len

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        return tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1],
                                    use_gpu=self.use_gpu)

    def forward(self, src_seq, src_synxs, tgt_seq=None, mode="train", **kwargs):

        if mode == "train":
            assert tgt_seq is not None

            tgt_seq = tgt_seq.transpose(1, 0).contiguous()  # length first

            return self.force_teaching(src_seq, src_synxs, tgt_seq, **kwargs)

        elif mode == "infer":
            return self.batch_beam_search(x=src_seq, synxs=src_synxs, **kwargs)
