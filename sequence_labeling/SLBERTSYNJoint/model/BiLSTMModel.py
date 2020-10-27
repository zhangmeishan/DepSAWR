from module.MyLSTM import *
from module.ScaleMix import *
from module.Utils import *
from module.CRF import *


class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, parser_config, input_dims, bert_layers):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.PAD = vocab.PAD
        self.input_dims = input_dims
        self.input_depth = bert_layers if config.bert_tune == 0 else 1
        self.hidden_dims = 2 * config.lstm_hiddens
        self.projections = nn.ModuleList([NonLinear(self.input_dims, self.hidden_dims, activation=GELU()) \
                                          for i in range(self.input_depth)])

        self.rescale = ScalarMix(mixture_size=self.input_depth)

        parser_dim = 2 * parser_config.lstm_hiddens
        self.transformer_lstm = nn.ModuleList(
            [NonLinear(parser_dim, self.hidden_dims, activation=GELU())
             for i in range(parser_config.lstm_layers)])

        parser_mlp_dim = parser_config.mlp_arc_size + parser_config.mlp_rel_size
        self.transformer_dep = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())
        self.transformer_head = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())

        self.parser_lstm_layers = parser_config.lstm_layers
        self.synscale = ScalarMix(mixture_size=3+parser_config.lstm_layers)

        self.predicate_embed = nn.Embedding(3, config.predict_dims, padding_idx=0)
        nn.init.normal_(self.predicate_embed.weight, 0.0, 1.0 / (config.predict_dims ** 0.5))

        self.lstm_input_dims = 2 * self.hidden_dims + config.predict_dims

        self.bilstm = MyLSTM(
            input_size=self.lstm_input_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.outlayer = nn.Linear(2 * config.lstm_hiddens, vocab.label_size, bias=False)
        nn.init.normal_(self.outlayer.weight, 0.0, 1.0 / ((2 * config.lstm_hiddens) ** 0.5))

        self.crf = CRF(vocab.label_size)

    def forward(self, inputs, predicts, synxs, masks):
        # x = (batch size, sequence length, dimension of embedding)
        proj_hiddens = []
        for idx, input in enumerate(inputs):
            cur_hidden = self.projections[idx](input)
            proj_hiddens.append(cur_hidden)

        x_embed = self.rescale(proj_hiddens)

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

        x_predict_embed = self.predicate_embed(predicts)

        if self.training:
            x_embed, x_syn, x_predict_embed = drop_tri_input_independent(
                x_embed, x_syn, x_predict_embed, self.config.dropout_emb)

        embeddings = torch.cat((x_embed, x_syn, x_predict_embed), dim=2)

        lstm_out, _ = self.bilstm(embeddings, masks)
        lstm_out = lstm_out.transpose(1, 0)

        label_scores = self.outlayer(lstm_out)

        return label_scores

    def compute_loss(self, output, answer, masks):
        # output: [B, T, L], answer: [B, T], mask: [B, T, L]
        # print answer
        output = output.transpose(1, 0).contiguous()
        answer = answer.transpose(1, 0).contiguous()
        masks = masks.transpose(1, 0).contiguous()
        total_loss = self.crf(output, answer, masks)

        num_words = masks.float().sum()
        total_loss = total_loss / num_words

        return total_loss

    def decode(self, label_scores, masks):
        label_scores = label_scores.transpose(1, 0).contiguous()
        masks = masks.transpose(1, 0).contiguous()
        tag_seq = self.crf.decode(label_scores, masks)

        return tag_seq

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        self.load_state_dict(torch.load(filepath))
        print('Loaded model from: {}'.format(filepath))