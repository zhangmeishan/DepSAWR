from module.Common import *


class NDLSTM(nn.Module):

    """single layer bilstm without any dropout"""

    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super(NDLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.fcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        if self.bidirectional:
            self.bcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fcell.weight_ih.data)
        nn.init.orthogonal_(self.fcell.weight_hh.data)
        nn.init.constant_(self.fcell.bias_ih.data, 0.0)
        nn.init.constant_(self.fcell.bias_hh.data, 0.0)
        hidden_size = self.fcell.bias_hh.data.shape[0] // 4
        self.fcell.bias_hh.data[hidden_size:(2*hidden_size)] = 1.0

        if self.bidirectional:
            nn.init.xavier_uniform_(self.bcell.weight_ih.data)
            nn.init.orthogonal_(self.bcell.weight_hh.data)
            nn.init.constant_(self.bcell.bias_ih.data, 0.0)
            nn.init.constant_(self.bcell.bias_hh.data, 0.0)
            hidden_size = self.bcell.bias_hh.data.shape[0] // 4
            self.bcell.bias_hh.data[hidden_size:(2 * hidden_size)] = 1.0

    @staticmethod
    def _forward_rnn(cell, input, masks, initial):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in range(max_time):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            hx = (h_next, c_next)
        output = torch.stack(output, 0)
        return output, hx

    @staticmethod
    def _forward_brnn(cell, input, masks, initial):
        max_time = input.size(0)
        output = []
        hx = initial
        for time in reversed(range(max_time)):
            h_next, c_next = cell(input=input[time], hx=hx)
            h_next = h_next*masks[time] + initial[0]*(1-masks[time])
            c_next = c_next*masks[time] + initial[1]*(1-masks[time])
            output.append(h_next)
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)


        layer_output, (layer_h, layer_c) = NDLSTM._forward_rnn(cell=self.fcell, \
            input=input, masks=masks, initial=initial)
        if self.bidirectional:
            blayer_output, (blayer_h, blayer_c) = NDLSTM._forward_brnn(cell=self.bcell, \
                input=input, masks=masks, initial=initial)

        hidden = torch.cat([layer_h, blayer_h], 1) if self.bidirectional else layer_h
        cell = torch.cat([layer_c, blayer_c], 1) if self.bidirectional else layer_c
        output = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (hidden, cell)
