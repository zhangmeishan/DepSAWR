from module.Utils import *
from module.MyLSTM import *
from module.ScaleMix import *
import torch.nn.functional as F


class SAModel(nn.Module):
    def __init__(self, vocab, config, parser_config, input_dims, bert_layers):
        super(SAModel, self).__init__()
        self.config = config
        self.input_dims = input_dims
        self.input_depth = bert_layers if config.bert_tune == 0 else 1
        self.hidden_dims = 2 * config.lstm_hiddens
        self.projections = nn.ModuleList(
            [NonLinear(self.input_dims, self.hidden_dims, activation=GELU())
             for i in range(self.input_depth)])

        self.rescale = ScalarMix(mixture_size=self.input_depth)

        self.transformer_emb = NonLinear(parser_config.word_dims, self.hidden_dims, activation=GELU())

        parser_dim = 2 * parser_config.lstm_hiddens
        self.transformer_lstm = nn.ModuleList(
            [NonLinear(parser_dim, self.hidden_dims, activation=GELU())
             for i in range(parser_config.lstm_layers)])

        parser_mlp_dim = parser_config.mlp_arc_size + parser_config.mlp_rel_size
        self.transformer_dep = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())
        self.transformer_head = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())

        self.parser_lstm_layers = parser_config.lstm_layers
        self.synscale = ScalarMix(mixture_size=3+parser_config.lstm_layers)

        self.lstm = MyLSTM(
            input_size=2*self.hidden_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2 * config.lstm_hiddens, vocab.tag_size, bias=False)

    def forward(self, inputs, synxs, masks):
        proj_hiddens = []
        for idx, input in enumerate(inputs):
            cur_hidden = self.projections[idx](input)
            proj_hiddens.append(cur_hidden)

        word_represents = self.rescale(proj_hiddens)

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
            word_represents, x_syn = drop_bi_input_independent(word_represents, x_syn, self.config.dropout_emb)

        x_lexical = torch.cat((word_represents, x_syn), dim=2)

        hiddens, _ = self.lstm(x_lexical, masks, None)
        hiddens = hiddens.transpose(1, 0)

        mask_values = (masks.unsqueeze(-1).expand(hiddens.size()) - 1) * float('1e6')
        hiddens = hiddens + mask_values
        hiddens = hiddens.permute(0, 2, 1)

        hidden = F.max_pool1d(hiddens, hiddens.size(2)).squeeze(2)
        outputs = self.proj(hidden)

        return outputs
