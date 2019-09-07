from module.MyLSTM import *
from module.Utils import *
from data.Vocab import *
from module.CPUEmbedding import *

class ParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(ParserModel, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=vocab.PAD)
        self.extword_embed = CPUEmbedding(vocab.extvocab_size, config.word_dims, padding_idx=vocab.PAD)

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
            x_embed = drop_uni_input_independent(x_embed, self.config.dropout_emb)

        outputs, (_, _, allhids) = self.lstm(x_embed, masks, None)

        for onehid in allhids:
            results.append(onehid.transpose(1, 0))

        outputs = outputs.transpose(1, 0)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        results.append(x_all_dep)
        results.append(x_all_head)

        return results
