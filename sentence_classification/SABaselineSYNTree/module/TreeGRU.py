import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from module.Tree import *
import numpy as np


class DTTreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(DTTreeGRU, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # linear parameters for transformation from input to hidden state. same for all 5 gates
        self.gate_ih = nn.Linear(in_features=input_size, out_features=5*hidden_size,bias=True)
        self.gate_lhh = nn.Linear(in_features=hidden_size, out_features=5*hidden_size,bias=False)
        self.gate_rhh = nn.Linear(in_features=hidden_size, out_features=5*hidden_size, bias=False)
        self.cell_ih = nn.Linear(in_features=input_size, out_features=hidden_size,bias=True)
        self.cell_lhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.cell_rhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, inputs, indexes, trees, lengths):
        """
        :param inputs: batch first
        :param tree:
        :return: output, h_n
        """

        max_length, batch_size, input_dim = inputs.size()
        dt_state = []
        degree = np.zeros((batch_size, max_length), dtype=np.int32)
        last_indexes = np.zeros((batch_size), dtype=np.int32)
        for b, tree in enumerate(trees):
            dt_state.append({})
            for index in range(lengths[b]):
                degree[b, index] = tree[index].left_num + tree[index].right_num

        zeros = inputs.data.new(self._hidden_size).fill_(0.)

        for step in range(max_length):
            step_inputs, left_child_hs, right_child_hs, compute_indexes = [], [], [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in range(last_index, lengths[b]):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] += 1
                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    if tree[cur_index].left_num == 0:
                        left_child_h = zeros
                    else:
                        left_child_h = [dt_state[b][child.index] for child in tree[cur_index].left_children]
                        left_child_h = torch.stack(left_child_h, 0)
                        left_child_h = torch.sum(left_child_h, dim=0)

                    if tree[cur_index].right_num == 0:
                        right_child_h = zeros
                    else:
                        right_child_h = [dt_state[b][child.index] for child in tree[cur_index].right_children]
                        right_child_h = torch.stack(right_child_h, 0)
                        right_child_h = torch.sum(right_child_h, dim=0)

                    left_child_hs.append(left_child_h)
                    right_child_hs.append(right_child_h)

            if len(compute_indexes) == 0:
                for b, last_index in enumerate(last_indexes):
                    if last_index != lengths[b]:
                        print('bug exists: some nodes are not completed')
                break

            step_inputs = torch.stack(step_inputs, 0)
            left_child_hs = torch.stack(left_child_hs, 0)
            right_child_hs = torch.stack(right_child_hs, 0)

            results = self.node_forward(step_inputs, left_child_hs, right_child_hs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                dt_state[b][cur_index] = results[idx]
                if trees[b][cur_index].parent is not None:
                    parent_index = trees[b][cur_index].parent.index
                    degree[b, parent_index] -= 1
                    if degree[b, parent_index] < 0:
                        print('strange bug')

        outputs, output_t = [], []

        for b in range(batch_size):
            output = [dt_state[b][idx] for idx in range(1,lengths[b])] + [dt_state[b][0]] \
                     + [zeros for idx in range(lengths[b], max_length)]
            outputs.append(torch.stack(output, 0))
            output_t.append(dt_state[b][0])

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def node_forward(self, input, left_child_h, right_child_h):
        gates = self.gate_ih(input) + self.gate_lhh(left_child_h) + self.gate_rhh(right_child_h)
        gates = torch.sigmoid(gates)
        rl, rr, zl, zr, z = torch.split(gates, gates.size(1) // 5, dim=1)

        gated_l,  gated_r = rl*left_child_h, rr*right_child_h
        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zl*left_child_h + zr*right_child_h + z*cell

        return hidden


class TDTreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(TDTreeGRU, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # linear parameters for transformation from input to hidden state. same for all 5 gates
        self.gate_ih = nn.Linear(in_features=input_size, out_features=3*hidden_size,bias=True)
        self.gate_lhh = nn.Linear(in_features=hidden_size, out_features=3*hidden_size,bias=False)
        self.gate_rhh = nn.Linear(in_features=hidden_size, out_features=3*hidden_size, bias=False)
        self.cell_ih = nn.Linear(in_features=input_size, out_features=hidden_size,bias=True)
        self.cell_lhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.cell_rhh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)


    def forward(self, inputs, indexes, trees, lengths):
        """
        :param inputs:
        :param tree:
        :return: output, h_n
        """
        max_length, batch_size, input_dim = inputs.size()
        degree = np.ones((batch_size, max_length), dtype=np.int32)
        last_indexes = max_length * np.ones((batch_size), dtype=np.int32)
        td_state = []
        for b in range(batch_size):
            td_state.append({})
            root_index = indexes[lengths[b] - 1, b]
            degree[b, root_index] = 0
            last_indexes[b] = lengths[b]

        zeros = inputs.data.new(self._hidden_size).fill_(0.)
        for step in range(max_length):
            step_inputs, left_parent_hs, right_parent_hs, compute_indexes = [], [], [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in reversed(range(last_index)):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] -= 1
                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    parent_h = zeros
                    if tree[cur_index].parent is None:
                        left_parent_hs.append(parent_h)
                        right_parent_hs.append(parent_h)
                    else:
                        valid_parent_h = td_state[b][tree[cur_index].parent.index]
                        if tree[cur_index].is_left:
                            left_parent_hs.append(valid_parent_h)
                            right_parent_hs.append(parent_h)
                        else:
                            left_parent_hs.append(parent_h)
                            right_parent_hs.append(valid_parent_h)

            if len(compute_indexes) == 0:
                for last_index in last_indexes:
                    if last_index != 0:
                        print('bug exists: some nodes are not completed')
                break

            step_inputs = torch.stack(step_inputs, 0)
            left_parent_hs = torch.stack(left_parent_hs, 0)
            right_parent_hs = torch.stack(right_parent_hs, 0)

            results = self.node_forward(step_inputs, left_parent_hs, right_parent_hs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                td_state[b][cur_index] = results[idx]
                for child in trees[b][cur_index].left_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')
                for child in trees[b][cur_index].right_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')

        outputs, output_t = [], []
        for b in range(batch_size):
            output = [td_state[b][idx] for idx in range(1,lengths[b])] + [td_state[b][0]] \
                     + [zeros for idx in range(lengths[b], max_length)]
            outputs.append(torch.stack(output, 0))
            output_t.append(td_state[b][0])

        return torch.stack(outputs, 0), torch.stack(output_t, 0)

    def node_forward(self, input, left_parent_h, right_parent_h):
        gates = self.gate_ih(input) + self.gate_lhh(left_parent_h) + self.gate_rhh(right_parent_h)
        gates = torch.sigmoid(gates)
        rp, zp, z = torch.split(gates, gates.size(1) // 3, dim=1)

        gated_l, gated_r = rp*left_parent_h,  rp*right_parent_h

        cell = self.cell_ih(input) + self.cell_lhh(gated_l) + self.cell_rhh(gated_r)
        cell = torch.tanh(cell)

        hidden = zp*(left_parent_h + right_parent_h) + z*cell

        return hidden