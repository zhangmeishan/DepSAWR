from module.MyLSTM import *
from module.Utils import *
from module.CRF import *


class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, elmo_shape):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.PAD = vocab.PAD
        self.word_dims = config.word_dims
        self.elmo_layers = elmo_shape[0]
        self.elmo_dims = elmo_shape[1]

        weights = torch.randn(self.elmo_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.mlp_elmo = nn.Linear(self.elmo_dims, self.word_dims, bias=False)

        self.predicate_embed = nn.Embedding(3, config.predict_dims, padding_idx=0)
        nn.init.normal_(self.predicate_embed.weight, 0.0, 1.0 / (config.predict_dims ** 0.5))

        self.lstm_input_dims = config.word_dims + config.predict_dims

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

    def forward(self, elmos, predicts, masks):
        # x = (batch size, sequence length, dimension of embedding)
        elmos = elmos.permute(0, 2, 3, 1).matmul(self.weights)
        x_embed = self.mlp_elmo(elmos)

        x_predict_embed = self.predicate_embed(predicts)

        if self.training:
            x_embed, x_predict_embed = drop_bi_input_independent(x_embed, x_predict_embed, self.config.dropout_emb)

        embeddings = torch.cat((x_embed, x_predict_embed), 2)

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