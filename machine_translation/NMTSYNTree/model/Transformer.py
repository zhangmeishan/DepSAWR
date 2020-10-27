import torch.nn as nn
import torch.nn.functional as F
from module.Basic import BottleLinear as Linear
from module.Sublayers import PositionwiseFeedForward, MultiHeadedAttention
from module.Embeddings import Embeddings
from module.Utils import *
from module.TreeGRU import *
from module.Tree import *


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('bool')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class Encoder(nn.Module):
    def __init__(
            self, src_vocab, rel_vocab,n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super().__init__()

        self.src_vocab = src_vocab
        self.d_word_vec = d_word_vec
        self.d_model = d_model

        self.num_layers = n_layers
        self.embeddings = Embeddings(num_embeddings=src_vocab.max_n_words,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     padding_idx=src_vocab.PAD,
                                     add_position_embedding=True
                                     )
        self.rel_vocab = rel_vocab
        self.rel_embedding = Embeddings(num_embeddings=rel_vocab.max_n_rels,
                                        embedding_dim=d_word_vec,
                                        dropout=0.0,
                                        add_position_embedding=False,
                                        padding_idx=rel_vocab.PAD)

        self.dt_tree = DTTreeGRU(2*d_word_vec, d_model)
        self.td_tree = TDTreeGRU(2*d_word_vec, d_model)
        self.transform = nn.Linear(in_features=2 * d_model, out_features=d_model, bias=True)

        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.extension_fuc = None

    def get_embeddings(self, seqs):
        emb = self.embeddings(seqs)
        return emb

    def forward(self, src_seq, rels, heads, xlengths):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.get_embeddings(src_seq)

        rel_emb = self.rel_embedding(rels)
        outputs = torch.cat([emb, rel_emb], 2)
        outputs = outputs.transpose(0, 1)
        max_length, batch_size, input_dim = outputs.size()

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree(outputs, indexes, trees, xlengths)
        td_outputs, td_hidden_ts = self.td_tree(outputs, indexes, trees, xlengths)

        tree_outputs = torch.cat([dt_outputs, td_outputs], dim=2)
        tree_hiddens = self.transform(tree_outputs)

        enc_mask = src_seq.data.eq(self.src_vocab.PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = tree_hiddens

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask


class DecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(all_input, all_input, input_norm,
                                                  mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
                                                  mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()

        self.tgt_vocab = tgt_vocab

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model
        self.d_word_vec = d_word_vec

        self.embeddings = Embeddings(tgt_vocab.max_n_words, d_word_vec,
                                     dropout=dropout,
                                     padding_idx=tgt_vocab.PAD,
                                     add_position_embedding=True)

        self.block_stack = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout)
            for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(d_model)
        self.extension_fuc = None

    @property
    def dim_per_head(self):
        return self.d_model // self.n_head

    def get_embeddings(self, seqs):
        emb = self.embeddings(seqs)

        return emb

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.get_embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(self.tgt_vocab.PAD).unsqueeze(1).expand(batch_size, query_len,
                                                                                            key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache \
                = self.block_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


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


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, config, src_vocab, rel_vocab, tgt_vocab, use_gpu=True):

        super(Transformer, self).__init__()

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_pad = src_vocab.PAD
        self.src_bos = src_vocab.BOS
        self.src_eos = src_vocab.EOS
        self.tgt_pad = tgt_vocab.PAD
        self.tgt_bos = tgt_vocab.BOS
        self.tgt_eos = tgt_vocab.EOS

        self.encoder = Encoder(
            src_vocab, rel_vocab, n_layers=config.num_layers, n_head=config.num_heads,
            d_word_vec=config.embed_size, d_model=config.embed_size,
            d_inner_hid=config.attention_size, dropout=config.dropout_hidden)

        self.decoder = Decoder(
            tgt_vocab, n_layers=config.num_layers, n_head=config.num_heads,
            d_word_vec=config.embed_size, d_model=config.embed_size,
            d_inner_hid=config.attention_size, dropout=config.dropout_hidden)

        self.dropout = nn.Dropout(config.dropout_hidden)

        if config.proj_share_weight:
            self.generator = Generator(vocab=self.tgt_vocab,
                                       hidden_size=config.embed_size,
                                       shared_weight=self.decoder.embeddings.embeddings.weight)

        else:
            self.generator = Generator(vocab=self.tgt_vocab, hidden_size=config.embed_size)

        self.use_gpu = use_gpu

    def forward(self, src_seq, src_rel, src_head, tgt_seq=None, mode="train", **kwargs):
        if mode == "train":
            assert tgt_seq is not None
            return self.force_teaching(src_seq, src_rel, src_head, tgt_seq, **kwargs)
        elif mode == "infer":
            return self.batch_beam_search(src_seq=src_seq, rel=src_rel, head=src_head, **kwargs)

    def force_teaching(self, src_seq, rel, head, tgt_seq, lengths):

        enc_output, enc_mask = self.encoder(src_seq, rel, head, lengths)
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)

        return self.generator(dec_output)

    def batch_beam_search(self, src_seq, rel, head, lengths, beam_size=5, max_steps=150):

        batch_size = src_seq.size(0)

        enc_output, enc_mask = self.encoder(src_seq, rel, head, lengths)  # [batch_size, seq_len, dim]

        # dec_caches = self.decoder.compute_caches(enc_output)

        # Tile beam_size times
        enc_mask = tile_batch(enc_mask, multiplier=beam_size, batch_dim=0)
        enc_output = tile_batch(enc_output, multiplier=beam_size, batch_dim=0)

        final_word_indices = src_seq.new(batch_size, beam_size, 1).fill_(self.tgt_bos)  # Word indices in the beam
        final_lengths = enc_output.new(batch_size, beam_size).fill_(0.0)  # length of the sentence
        beam_mask = enc_output.new(batch_size, beam_size).fill_(1.0)  # Mask of beams
        beam_scores = enc_output.new(batch_size, beam_size).fill_(0.0)  # Accumulated scores of the beam

        self_attn_caches = None  # Every element has shape [batch_size * beam_size, num_heads, seq_len, dim_head]
        enc_attn_caches = None

        for t in range(max_steps):

            inp_t = final_word_indices.view(-1, final_word_indices.size(-1))

            dec_output, self_attn_caches, enc_attn_caches \
                = self.decoder(tgt_seq=inp_t,
                               enc_output=enc_output,
                               enc_mask=enc_mask,
                               enc_attn_caches=enc_attn_caches,
                               self_attn_caches=self_attn_caches)  # [batch_size * beam_size, seq_len, dim]

            next_scores = - self.generator(dec_output[:, -1].contiguous()).data  # [batch_size * beam_size, n_words]
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

            # Re-arrange by new beam indices
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

            final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=final_lengths,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1],
                                                 use_gpu=self.use_gpu)

            self_attn_caches = map_structure(
                lambda t: tensor_gather_helper(gather_indices=next_beam_ids,
                                               gather_from=t.data,
                                               batch_size=batch_size,
                                               beam_size=beam_size,
                                               gather_shape=[batch_size * beam_size, self.decoder.n_head,
                                                             -1, self.decoder.dim_per_head],
                                               use_gpu=self.use_gpu), self_attn_caches)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(self.tgt_eos).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                       self.tgt_pad)
            # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # # If an EOS or PAD is encountered, set the beam mask to 0.0
            # beam_mask_ = next_word_ids.gt(Vocab.EOS).float()
            # beam_mask = beam_mask * beam_mask_

            final_lengths += beam_mask

            final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

            if beam_mask.eq(0.0).all():
                break

        scores = beam_scores / final_lengths

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        return tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1],
                                    use_gpu=self.use_gpu)
