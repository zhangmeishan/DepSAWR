from module.CPUEmbedding import *

class ExtWord(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(ExtWord, self).__init__()
        self.config = config
        extvocab_size, extword_dims = pretrained_embedding.shape
        self.word_dims = extword_dims

        if config.word_dims != extword_dims or vocab.extvocab_size != extvocab_size:
            print("word vocab size does not match, check word embedding file")
        self.extword_embed = CPUEmbedding(vocab.extvocab_size, self.word_dims, padding_idx=vocab.PAD)
        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

    def forward(self, extwords):

        return self.extword_embed(extwords)


