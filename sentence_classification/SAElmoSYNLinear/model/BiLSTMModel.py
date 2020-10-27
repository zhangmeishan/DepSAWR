from module.MyLSTM import *
from module.Utils import *
import torch.nn.functional as F


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

        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        word_init = np.random.randn(vocab.vocab_size, config.word_dims).astype(np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        self.lstm = MyLSTM(
            input_size=self.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.proj = nn.Linear(2 * config.lstm_hiddens, vocab.tag_size, bias=False)

    def forward(self, elmos, actions, masks):
        # x = (batch size, sequence length, dimension of embedding)
        # input elmos: (batch size, layers, length, dimensions)
        elmos = elmos.matmul(self.weights)
        x_elmo_embed = self.mlp_elmo(elmos)
        x_action_embed = self.word_embed(actions)
        x_embed = x_elmo_embed + x_action_embed

        if self.training:
            x_embed = drop_uni_input_independent(x_embed, self.config.dropout_emb)

        hiddens, _ = self.lstm(x_embed, masks, None)
        hiddens = hiddens.transpose(1, 0)

        mask_values = (masks.unsqueeze(-1).expand(hiddens.size()) - 1) * float('1e6')
        hiddens = hiddens + mask_values
        hiddens = hiddens.permute(0, 2, 1)

        hidden = F.max_pool1d(hiddens, hiddens.size(2)).squeeze(2)
        outputs = self.proj(hidden)

        return outputs
