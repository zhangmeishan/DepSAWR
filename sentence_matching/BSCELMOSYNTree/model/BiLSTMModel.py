from module.ESIM import *
from module.Utils import *
from module.CPUEmbedding import *
from module.Common import *
from module.TreeGRU import *

class BiLSTMModel(nn.Module):
    def __init__(self, vocab, config, init_embedding, elmo_shape):
        super(BiLSTMModel, self).__init__()
        self.config = config
        self.elmo_layers = elmo_shape[0]
        self.elmo_dims = elmo_shape[1]
        initvocab_size, initword_dims = init_embedding.shape
        self.word_dims = initword_dims

        weights = torch.randn(self.elmo_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.mlp_elmo = nn.Linear(self.elmo_dims, self.word_dims)

        if config.word_dims != initword_dims or vocab.vocab_size != initvocab_size:
            print("prev embedding shape size does not match, check config file")
        self.word_embed = nn.Embedding(vocab.vocab_size, self.word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(init_embedding))

        self.rel_embed = nn.Embedding(vocab.rel_size, self.word_dims, padding_idx=vocab.PAD)

        embedding_matrix = np.zeros((vocab.rel_size, self.word_dims))
        for i in range(vocab.rel_size):
            if i == vocab.PAD: continue
            embedding_matrix[i] = np.random.normal(size=(self.word_dims))
            embedding_matrix[i] = embedding_matrix[i] / np.std(embedding_matrix[i])
        self.rel_embed.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.rnn_dropout = RNNDropout(p=config.dropout_mlp)

        self.dt_tree = DTTreeGRU(3*self.word_dims, config.lstm_hiddens)
        self.td_tree = TDTreeGRU(3*self.word_dims, config.lstm_hiddens)

        self.hidden_size = config.lstm_hiddens
        self.lstm_enc = Seq2SeqEncoder(nn.LSTM,
                                    2*self.hidden_size,
                                    self.hidden_size,
                                    bidirectional=True)

        self.atten = SoftmaxAttention()

        self.hidden_dim = 4*2*config.lstm_hiddens

        self.mlp = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                        self.hidden_size),
                                        nn.ReLU())

        self.lstm_dec = Seq2SeqEncoder(nn.LSTM,
                                    self.hidden_size,
                                    self.hidden_size,
                                    bidirectional=True)

        self.feature_dim = 2*4*config.lstm_hiddens
        self.proj = nn.Sequential(nn.Dropout(p=config.dropout_mlp),
                                nn.Linear(2*4*self.hidden_size,  self.hidden_size),
                                nn.Tanh(),
                                nn.Dropout(p=config.dropout_mlp),
                                nn.Linear(self.hidden_size, vocab.tag_size))

        self.apply(_init_esim_weights)


    def forward(self, tinputs):
        ##unpack inputs
        src_words, src_elmos, src_rels, src_heads, src_lens, src_masks, \
        tgt_words, tgt_elmos, tgt_rels, tgt_heads, tgt_lens, tgt_masks = tinputs

        src_elmos = src_elmos.permute(0, 2, 3, 1).matmul(self.weights)
        src_elmo_embed = self.mlp_elmo(src_elmos)

        tgt_elmos = tgt_elmos.permute(0, 2, 3, 1).matmul(self.weights)
        tgt_elmo_embed = self.mlp_elmo(tgt_elmos)

        src_dyn_embed = self.word_embed(src_words)
        tgt_dyn_embed = self.word_embed(tgt_words)

        src_rel_embed = self.rel_embed(src_rels)
        tgt_rel_embed = self.rel_embed(tgt_rels)

        src_embed = torch.cat([src_dyn_embed, src_elmo_embed, src_rel_embed], dim=-1)
        tgt_embed = torch.cat([tgt_dyn_embed, tgt_elmo_embed, tgt_rel_embed], dim=-1)

        src_embed = self.rnn_dropout(src_embed)
        tgt_embed = self.rnn_dropout(tgt_embed)

        batch_size, src_length, input_dim = src_embed.size()
        src_trees = []
        src_indexes = np.zeros((src_length, batch_size), dtype=np.int32)
        for b, head in enumerate(src_heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                src_indexes[step, b] = index
            src_trees.append(tree)
        src_embed = src_embed.transpose(1, 0)
        src_dt_outputs, src_dt_hidden_ts = self.dt_tree(src_embed, src_indexes, src_trees, src_lens)
        src_td_outputs, src_td_hidden_ts = self.td_tree(src_embed, src_indexes, src_trees, src_lens)
        src_shiddens = torch.cat([src_dt_outputs, src_td_outputs], dim=2)

        batch_size, tgt_length, input_dim = tgt_embed.size()
        tgt_trees = []
        tgt_indexes = np.zeros((tgt_length, batch_size), dtype=np.int32)
        for b, head in enumerate(tgt_heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                tgt_indexes[step, b] = index
            tgt_trees.append(tree)
        tgt_embed = tgt_embed.transpose(1, 0)
        tgt_dt_outputs, tgt_dt_hidden_ts = self.dt_tree(tgt_embed, tgt_indexes, tgt_trees, tgt_lens)
        tgt_td_outputs, tgt_td_hidden_ts = self.td_tree(tgt_embed, tgt_indexes, tgt_trees, tgt_lens)
        tgt_shiddens = torch.cat([tgt_dt_outputs, tgt_td_outputs], dim=2)

        src_hiddens = self.lstm_enc(src_shiddens, src_lens)
        tgt_hiddens = self.lstm_enc(tgt_shiddens, tgt_lens)

        src_hiddens_att, tgt_hiddens_att = self.atten(src_hiddens, src_masks, \
                                                      tgt_hiddens, tgt_masks)

        src_diff_hiddens = src_hiddens - src_hiddens_att
        src_prod_hiddens = src_hiddens * src_hiddens_att
        src_summary_hiddens = torch.cat([src_hiddens, src_hiddens_att, src_diff_hiddens, \
                                         src_prod_hiddens], dim=-1)

        tgt_diff_hiddens = tgt_hiddens - tgt_hiddens_att
        tgt_prod_hiddens = tgt_hiddens * tgt_hiddens_att
        tgt_summary_hiddens = torch.cat([tgt_hiddens, tgt_hiddens_att, tgt_diff_hiddens, \
                                         tgt_prod_hiddens], dim=-1)

        src_hiddens_proj = self.mlp(src_summary_hiddens)
        tgt_hiddens_proj = self.mlp(tgt_summary_hiddens)

        src_hiddens_proj = self.rnn_dropout(src_hiddens_proj)
        tgt_hiddens_proj = self.rnn_dropout(tgt_hiddens_proj)

        src_final_hiddens = self.lstm_dec(src_hiddens_proj, src_lens)
        tgt_final_hiddens = self.lstm_dec(tgt_hiddens_proj, tgt_lens)

        src_hidden_avg = torch.sum(src_final_hiddens * src_masks.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / (torch.sum(src_masks, dim=1, keepdim=True) + 1e-7)
        tgt_hidden_avg = torch.sum(tgt_final_hiddens * tgt_masks.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / (torch.sum(tgt_masks, dim=1, keepdim=True) + 1e-7)

        src_hidden_max, _ = replace_masked(src_final_hiddens, src_masks, -1e7).max(dim=1)
        tgt_hidden_max, _ = replace_masked(tgt_final_hiddens, tgt_masks, -1e7).max(dim=1)

        hiddens = torch.cat([src_hidden_avg, src_hidden_max, tgt_hidden_avg, tgt_hidden_max], dim=1)

        outputs = self.proj(hiddens)
        return outputs


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0