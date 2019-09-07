import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from module import Init

class RNN(nn.Module):

    def __init__(self, type, batch_first=False, **kwargs):

        super().__init__()

        self.type = type
        self.batch_first = batch_first

        if self.type == "gru":
            self.rnn = nn.GRU(batch_first=batch_first, **kwargs)
        elif self.type == "lstm":
            self.rnn = nn.LSTM(batch_first=batch_first, **kwargs)

        self._reset_parameters()

    @property
    def batch_dim(self):
        if self.batch_first:
            return 0
        else:
            return 1

    def _reset_parameters(self):
        for weight in self.rnn.parameters():
            Init.rnn_init(weight.data)

    def forward(self, input, lengths, h_0=None):
        input_packed = pack_padded_sequence(input, lengths=lengths, batch_first=self.batch_first)
        out_packed, h_n = self.rnn(input_packed, h_0)
        out = pad_packed_sequence(out_packed, batch_first=self.batch_first)[0]
        return out.contiguous(), h_n
