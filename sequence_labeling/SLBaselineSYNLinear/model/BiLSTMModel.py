from module.MyLSTM import *
from module.Utils import *
from module.CPUEmbedding import *
from module.CRF import *


class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.PAD = vocab.PAD
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

    def forward(self, words, extwords, predicts, masks, indices):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed

        x_predict_embed = self.predicate_embed(predicts)

        if self.training:
            x_embed, x_predict_embed = drop_bi_input_independent(x_embed, x_predict_embed, self.config.dropout_emb)

        embeddings = torch.cat((x_embed, x_predict_embed), dim=2)

        lstm_out, _ = self.bilstm(embeddings, masks)
        lstm_out = lstm_out.transpose(1, 0)

        filtered = torch.gather(lstm_out, 1, indices)

        label_scores = self.outlayer(filtered)

        return label_scores

    def compute_loss(self, output, answer, wmasks):
        # output: [B, T, L], answer: [B, T], mask: [B, T, L]
        # print answer
        output = output.transpose(1, 0).contiguous()
        answer = answer.transpose(1, 0).contiguous()
        wmasks = wmasks.transpose(1, 0).contiguous()
        total_loss = self.crf(output, answer, wmasks)

        num_words = wmasks.float().sum()
        total_loss = total_loss / num_words

        return total_loss

    def decode(self, label_scores, wmasks):
        label_scores = label_scores.transpose(1, 0).contiguous()
        wmasks = wmasks.transpose(1, 0).contiguous()
        tag_seq = self.crf.decode(label_scores, wmasks)

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