from module.MyLSTM import *
from module.CPUEmbedding import *
import torch.nn.functional as F
from data.Vocab import *


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
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
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
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
        if vocab.extvocab_size != extvocab_size:
            print("word vocab size does not match, check word embedding file")
        self.extword_embed = CPUEmbedding(vocab.extvocab_size, self.word_dims, padding_idx=vocab.PAD)

        word_init = np.zeros((vocab.vocab_size, self.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))
        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.lstm = MyLSTM(
            input_size=self.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2*config.lstm_hiddens, vocab.tag_size, bias=False)

    def forward(self, words, extwords, masks):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed

        if self.training:
            x_embed = drop_input_independent(x_embed, self.config.dropout_emb)

        hiddens, _ = self.lstm(x_embed, masks, None)
        hiddens = hiddens.transpose(1, 0)

        mask_values = (masks.unsqueeze(-1).expand(hiddens.size()) - 1) * float('1e6')
        hiddens = hiddens + mask_values
        hiddens = hiddens.permute(0, 2, 1)

        hidden = F.max_pool1d(hiddens, hiddens.size(2)).squeeze(2)
        outputs = self.proj(hidden)

        return outputs
