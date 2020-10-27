import torch


class TensorInstances:
    def __init__(self, batch_size, slen, tlen, sblen, tblen):
        self.src_bert_indices = torch.zeros([batch_size, sblen], dtype=torch.int64, requires_grad=False)
        self.src_bert_segments = torch.zeros([batch_size, sblen], dtype=torch.int64, requires_grad=False)
        self.src_bert_pieces = torch.zeros([batch_size, slen, sblen], dtype=torch.float, requires_grad=False)
        self.src_masks = torch.zeros([batch_size, slen], dtype=torch.float, requires_grad=False)
        self.src_lens = torch.zeros([batch_size], dtype=torch.long, requires_grad=False)

        self.tgt_bert_indices = torch.zeros([batch_size, tblen], dtype=torch.int64, requires_grad=False)
        self.tgt_bert_segments = torch.zeros([batch_size, tblen], dtype=torch.int64, requires_grad=False)
        self.tgt_bert_pieces = torch.zeros([batch_size, tlen, tblen], dtype=torch.float, requires_grad=False)
        self.tgt_masks = torch.zeros([batch_size, tlen], dtype=torch.float, requires_grad=False)
        self.tgt_lens = torch.zeros([batch_size], dtype=torch.long, requires_grad=False)

        self.tags = torch.zeros([batch_size], dtype=torch.long, requires_grad=False)

    def to_cuda(self, device):
        self.src_bert_indices = self.src_bert_indices.cuda(device)
        self.src_bert_segments = self.src_bert_segments.cuda(device)
        self.src_bert_pieces = self.src_bert_pieces.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.src_lens = self.src_lens.cuda(device)

        self.tgt_bert_indices = self.tgt_bert_indices.cuda(device)
        self.tgt_bert_segments = self.tgt_bert_segments.cuda(device)
        self.tgt_bert_pieces = self.tgt_bert_pieces.cuda(device)
        self.tgt_masks = self.tgt_masks.cuda(device)
        self.tgt_lens = self.tgt_lens.cuda(device)

        self.tags = self.tags.cuda(device)

    @property
    def inputs(self):
        return self.src_bert_indices, self.src_bert_segments, self.src_bert_pieces,\
               self.tgt_bert_indices, self.tgt_bert_segments, self.tgt_bert_pieces, \
               self.src_lens, self.src_masks, self.tgt_lens, self.tgt_masks

    @property
    def outputs(self):
        return self.tags
