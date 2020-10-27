import torch


class TensorInstances:
    def __init__(self, batch_size, slen, tlen):
        self.src_words = torch.zeros([batch_size, slen], dtype=torch.int64, requires_grad=False)
        self.src_rels = torch.zeros([batch_size, slen], dtype=torch.int64, requires_grad=False)
        self.src_lens = torch.zeros(size=(batch_size,), requires_grad=False, dtype=torch.long)
        self.src_masks = torch.zeros([batch_size, slen], dtype=torch.float, requires_grad=False)
        self.tgt_words = torch.zeros([batch_size, tlen], dtype=torch.int64, requires_grad=False)
        self.tgt_rels = torch.zeros([batch_size, tlen], dtype=torch.int64, requires_grad=False)
        self.tgt_lens = torch.zeros(size=(batch_size,), requires_grad=False, dtype=torch.long)
        self.tgt_masks = torch.zeros([batch_size, tlen], dtype=torch.float, requires_grad=False)
        self.tags = torch.zeros(size=(batch_size,), requires_grad=False, dtype=torch.long)
        self.src_heads, self.tgt_heads = [], []

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.src_rels = self.src_rels.cuda(device)
        self.src_lens = self.src_lens.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tgt_words = self.tgt_words.cuda(device)
        self.tgt_rels = self.tgt_rels.cuda(device)
        self.tgt_lens = self.tgt_lens.cuda(device)
        self.tgt_masks = self.tgt_masks.cuda(device)
        self.tags = self.tags.cuda(device)

    @property
    def inputs(self):
        return (self.src_words, self.src_rels, self.src_heads, \
                self.src_lens, self.src_masks, \
                self.tgt_words, self.tgt_rels, self.tgt_heads, \
                self.tgt_lens, self.tgt_masks)

    @property
    def outputs(self):
        return self.tags
