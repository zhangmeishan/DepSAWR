from module.MyLSTM import *
from module.Utils import *
from module.TreeGRU import *
import torch.nn.functional as F
from data.Vocab import *


class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, elmo_shape):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.word_dims = config.word_dims
        self.elmo_layers = elmo_shape[0]
        self.elmo_dims = elmo_shape[1]

        weights = torch.randn(self.elmo_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.mlp_elmo = nn.Linear(self.elmo_dims, self.word_dims, bias=False)
        self.rel_embed = nn.Embedding(vocab.rel_size, self.word_dims, padding_idx=vocab.PAD)
        rel_init = np.random.randn(vocab.rel_size, config.word_dims).astype(np.float32)
        self.rel_embed.weight.data.copy_(torch.from_numpy(rel_init))

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

        self.proj = nn.Linear(2*config.lstm_hiddens, vocab.tag_size, bias=False)

    def forward(self, elmos, rels, heads, masks, lengths):
        # x = (batch size, sequence length, dimension of embedding)
        elmos = elmos.permute(0, 2, 3, 1).matmul(self.weights)
        x_embed = self.mlp_elmo(elmos)
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
