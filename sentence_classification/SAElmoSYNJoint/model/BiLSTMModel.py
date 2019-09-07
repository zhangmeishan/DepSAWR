from module.MyLSTM import *
from module.ScaleMix import *
from module.Utils import *
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, parser_config, elmo_shape):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.word_dims = config.word_dims
        self.elmo_layers = elmo_shape[0]
        self.elmo_dims = elmo_shape[1]

        weights = torch.randn(self.elmo_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.mlp_elmo = nn.Linear(self.elmo_dims, self.word_dims, bias=False)

        self.transformer_emb = nn.Linear(parser_config.word_dims, self.word_dims, bias=False)

        parser_dim = 2 * parser_config.lstm_hiddens
        self.transformer_lstm = []
        for layer in range(parser_config.lstm_layers):
            self.transformer_lstm.append(nn.Linear(parser_dim, self.word_dims, bias=False))

        parser_mlp_dim = parser_config.mlp_arc_size + parser_config.mlp_rel_size
        self.transformer_dep = nn.Linear(parser_mlp_dim, self.word_dims, bias=False)
        self.transformer_head = nn.Linear(parser_mlp_dim, self.word_dims, bias=False)

        self.parser_lstm_layers = parser_config.lstm_layers
        self.synscale = ScalarMix(mixture_size=3+parser_config.lstm_layers)

        self.lstm = MyLSTM(
            input_size=2*self.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2 * config.lstm_hiddens, vocab.tag_size, bias=False)


    def forward(self, elmos, synxs, masks):
        # x = (batch size, sequence length, dimension of embedding)
        elmos = elmos.transpose(-2, -1).matmul(self.weights)
        x_elmo = self.mlp_elmo(elmos)

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

        x_syn = self.synscale(x_syns)

        if self.training:
            x_elmo, x_syn = drop_bi_input_independent(x_elmo, x_syn, self.config.dropout_emb)

        x_lexical = torch.cat((x_elmo, x_syn), dim=2)

        hiddens, _ = self.lstm(x_lexical, masks, None)
        hiddens = hiddens.transpose(1, 0)

        mask_values = (masks.unsqueeze(-1).expand(hiddens.size()) - 1) * float('1e6')
        hiddens = hiddens  + mask_values
        hiddens = hiddens.permute(0, 2, 1)

        hidden = F.max_pool1d(hiddens, hiddens.size(2)).squeeze(2)
        outputs = self.proj(hidden)

        return outputs
