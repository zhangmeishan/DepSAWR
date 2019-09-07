from module.MyLSTM import *
from module.CPUEmbedding import *
from module.TreeGRU import *
import torch.nn.functional as F
from data.Vocab import *

def drop_tri_input_independent(word_embeddings, tag_embeddings, context_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    context_masks = context_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    context_masks = torch.bernoulli(context_masks)
    scale = 3.0 / (word_masks + tag_masks + context_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    context_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    context_masks = context_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    context_embeddings = context_embeddings * context_masks

    return word_embeddings, tag_embeddings, context_embeddings

def drop_bi_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    scale = 2.0 / (word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_uni_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.new_full((batch_size, hidden_size), 1 - dropout)
    drop_masks = torch.bernoulli(drop_masks)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(BiLSTMModel, self).__init__()
        self.config = config
        extvocab_size, extword_dims = pretrained_embedding.shape
        self.word_dims = extword_dims
        if config.word_dims != extword_dims:
            print("word dim size does not match, check config file")
        self.word_embed = nn.Embedding(vocab.vocab_size, self.word_dims, padding_idx=vocab.PAD)
        self.rel_embed = nn.Embedding(vocab.rel_size, self.word_dims, padding_idx=vocab.PAD)
        if vocab.extvocab_size != extvocab_size:
            print("word vocab size does not match, check word embedding file")
        self.extword_embed = CPUEmbedding(vocab.extvocab_size, self.word_dims, padding_idx=vocab.PAD)

        word_init = np.zeros((vocab.vocab_size, self.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))
        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.dt_tree = DTTreeGRU(2*self.word_dims, config.lstm_hiddens)
        self.td_tree = TDTreeGRU(2*self.word_dims, config.lstm_hiddens)

        self.lstm = MyLSTM(
            input_size=2*config.lstm_hiddens,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2 * config.lstm_hiddens, vocab.tag_size, bias=True)

    def forward(self, words, extwords, rels, heads, masks, lengths):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed

        x_rel_embed = self.rel_embed(rels)

        if self.training:
            x_embed, x_rel_embed = drop_bi_input_independent(x_embed, x_rel_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_rel_embed), dim=2)
        x_lexical = x_lexical.transpose(1, 0)

        max_length, batch_size, input_dim = x_lexical.size()

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree(x_lexical, indexes, trees, lengths)
        td_outputs, td_hidden_ts = self.td_tree(x_lexical, indexes, trees, lengths)

        tree_outputs = torch.cat([dt_outputs, td_outputs], dim=2)

        hiddens, _ = self.lstm(tree_outputs, masks, None)
        hiddens = hiddens.transpose(1, 0)

        mask_values = (masks.unsqueeze(-1).expand(hiddens.size()) - 1) * float('1e6')
        hiddens = hiddens + mask_values
        hiddens = hiddens.permute(0, 2, 1)

        hidden = F.max_pool1d(hiddens, hiddens.size(2)).squeeze(2)
        outputs = self.proj(hidden)

        return outputs
