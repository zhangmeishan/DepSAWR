from module.Utils import *
from module.MyLSTM import *
from module.ScaleMix import *
from module.TreeGRU import *
import torch.nn.functional as F
from data.Vocab import *


class SAModel(nn.Module):
    def __init__(self, vocab, config, input_dims, bert_layers):
        super(SAModel, self).__init__()
        self.config = config
        self.input_dims = input_dims
        self.input_depth = bert_layers if config.bert_tune == 0 else 1
        self.hidden_dims = 2 * config.lstm_hiddens
        self.projections = nn.ModuleList([NonLinear(self.input_dims, self.hidden_dims, activation=GELU()) \
                                          for i in range(self.input_depth)])
        self.rescale = ScalarMix(mixture_size=self.input_depth)

        self.rel_embed = nn.Embedding(vocab.rel_size, self.word_dims, padding_idx=vocab.PAD)
        rel_init = np.random.randn(vocab.rel_size, config.word_dims).astype(np.float32)
        self.rel_embed.weight.data.copy_(torch.from_numpy(rel_init))

        self.dt_tree = DTTreeGRU(2 * self.word_dims, config.lstm_hiddens)
        self.td_tree = TDTreeGRU(2 * self.word_dims, config.lstm_hiddens)

        self.lstm = MyLSTM(
            input_size=2 * config.lstm_hiddens,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2 * config.lstm_hiddens, vocab.tag_size, bias=False)

    def forward(self, inputs, rels, heads, masks, lengths):
        # x = (batch size, sequence length, dimension of embedding)
        proj_hiddens = []
        for idx, input in enumerate(inputs):
            cur_hidden = self.projections[idx](input)
            proj_hiddens.append(cur_hidden)

        x_embed = self.rescale(proj_hiddens)

        if self.training:
            x_embed = drop_uni_input_independent(x_embed, self.config.dropout_emb)

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
