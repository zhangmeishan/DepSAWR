import torch
from torch.autograd import Variable


class TensorInstances:
    def __init__(self, batch_size, slen, tlen):
        self.src_words = Variable(torch.LongTensor(batch_size, slen).zero_(), requires_grad=False)
        self.src_extwords = Variable(torch.LongTensor(batch_size, slen).zero_(), requires_grad=False)
        self.src_lens = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
        self.src_masks = Variable(torch.Tensor(batch_size, slen).zero_(), requires_grad=False)
        self.tgt_words = Variable(torch.LongTensor(batch_size, tlen).zero_(), requires_grad=False)
        self.tgt_extwords = Variable(torch.LongTensor(batch_size, tlen).zero_(), requires_grad=False)
        self.tgt_lens = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
        self.tgt_masks = Variable(torch.Tensor(batch_size, tlen).zero_(), requires_grad=False)
        self.tags = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

        self.src_dep_words = Variable(torch.LongTensor(batch_size, slen+1).zero_(), requires_grad=False)
        self.src_dep_extwords = Variable(torch.LongTensor(batch_size, slen+1).zero_(), requires_grad=False)
        self.src_dep_masks = Variable(torch.Tensor(batch_size, slen+1).zero_(), requires_grad=False)
        self.tgt_dep_words = Variable(torch.LongTensor(batch_size, tlen+1).zero_(), requires_grad=False)
        self.tgt_dep_extwords = Variable(torch.LongTensor(batch_size, tlen+1).zero_(), requires_grad=False)
        self.tgt_dep_masks = Variable(torch.Tensor(batch_size, tlen+1).zero_(), requires_grad=False)



    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.src_extwords = self.src_extwords.cuda(device)
        self.src_lens = self.src_lens.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tgt_words = self.tgt_words.cuda(device)
        self.tgt_extwords = self.tgt_extwords.cuda(device)
        self.tgt_lens = self.tgt_lens.cuda(device)
        self.tgt_masks = self.tgt_masks.cuda(device)
        self.tags = self.tags.cuda(device)

        self.src_dep_words = self.src_dep_words.cuda(device)
        self.src_dep_extwords = self.src_dep_extwords.cuda(device)
        self.src_dep_masks = self.src_dep_masks.cuda(device)
        self.tgt_dep_words = self.tgt_dep_words.cuda(device)
        self.tgt_dep_extwords = self.tgt_dep_extwords.cuda(device)
        self.tgt_dep_masks = self.tgt_dep_masks.cuda(device)

    @property
    def inputs(self):
        return (self.src_words, self.src_extwords, self.src_lens, self.src_masks, \
                self.tgt_words, self.tgt_extwords, self.tgt_lens, self.tgt_masks)

    @property
    def depinputs(self):
        return (self.src_dep_words, self.src_dep_extwords, self.src_dep_masks, \
                self.tgt_dep_words, self.tgt_dep_extwords, self.tgt_dep_masks)

    @property
    def outputs(self):
        return self.tags
