from module.Layer import *
from data.Vocab import *


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
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
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class ParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(ParserModel, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=vocab.PAD)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=vocab.PAD)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False


        self.lstm = MyLSTM(
            input_size=config.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.mlp_arc_dep = NonLinear(
            input_size = 2*config.lstm_hiddens,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.Tanh())
        self.mlp_arc_head = NonLinear(
            input_size = 2*config.lstm_hiddens,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.Tanh())

        self.total_num = int((config.mlp_arc_size+config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, \
                                     vocab.rel_size, bias=(True, True))
        self.arc_biaffine.linear.weight.requires_grad = False
        self.rel_biaffine.linear.weight.requires_grad = False


    def forward(self, words, extwords, masks):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed

        results = []
        results.append(x_embed)

        if self.training:
            x_embed = drop_sequence_sharedmask(x_embed, self.config.dropout_emb)

        outputs, (_, _, allhids) = self.lstm(x_embed, masks, None)

        for onehid in allhids:
            results.append(onehid.transpose(1, 0))

        outputs = outputs.transpose(1, 0)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        results.append(x_all_dep)
        results.append(x_all_head)

        return results
