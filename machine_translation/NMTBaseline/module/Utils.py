import torch
import numpy as np
import time
import math
import sys
import os

_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


def tile_batch(x, multiplier, batch_dim=0):
    """
    :type x: Variable
    """
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim + 1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled


def mask_scores(scores, beam_mask, eos_id):
    """
    Mask scores of next step according to beam mask.
    Args:
        scores (torch.Tensor): Scores of next tokens with shape [batch_size, beam_size, vocab_size].
            Smaller should be better (usually negative log-probs).
        beam_mask (torch.Tensor): Mask of beam. 1.0 means not closed and vice verse. The shape is
            [batch_size, beam_size]

    Returns:
        Masked scores of next tokens.
    """
    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[eos_id] = 0.0

    scores = scores * beam_mask.unsqueeze(2) + \
             torch.matmul((1.0 - beam_mask).unsqueeze(2), finished_row.unsqueeze(0))

    return scores


def tensor_gather_helper(gather_indices,
                         gather_from,
                         batch_size,
                         beam_size,
                         gather_shape,
                         use_gpu):
    range_ = (torch.arange(0, batch_size) * beam_size).long()

    if use_gpu:
        range_ = range_.cuda()

    gather_indices_ = (gather_indices + torch.unsqueeze(range_, 1)).view(-1)

    output = torch.index_select(gather_from.view(*gather_shape), 0, gather_indices_)

    out_size = gather_from.size()[:1 + len(gather_shape)]

    return output.view(*out_size)


def _yield_value(iterable):
    for value in iterable:
        yield value


def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


def is_sequence(seq):
    return isinstance(seq, list)


def flatten(nest):
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]


def _packed_nest_with_indices(structure, flat, index):
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(child)
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")

    flat_structure = flatten(structure)

    if len(flat_structure) != len(flat_sequence):
        raise ValueError("Count not pack sequence: expected {0} but got {1}".format(len(flat_structure),
                                                                                    len(flat_structure)))

    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)

    return packed


def _recursive_assert_same_structure(nest1, nest2):
    """Helper function for `assert_same_structure`."""
    is_sequence_nest1 = is_sequence(nest1)
    if is_sequence_nest1 != is_sequence(nest2):
        raise ValueError(
            "The two structures don't have the same nested structure.\n\n"
            "First structure: %s\n\nSecond structure: %s." % (nest1, nest2))

    if not is_sequence_nest1:
        return  # finished checking

    nest1_as_sequence = [n for n in _yield_value(nest1)]
    nest2_as_sequence = [n for n in _yield_value(nest2)]
    for n1, n2 in zip(nest1_as_sequence, nest2_as_sequence):
        _recursive_assert_same_structure(n1, n2)


def assert_same_structure(nest1, nest2):
    len_nest1 = len(flatten(nest1)) if is_sequence(nest1) else 1
    len_nest2 = len(flatten(nest2)) if is_sequence(nest2) else 1
    if len_nest1 != len_nest2:
        raise ValueError("The two structures don't have the same number of "
                         "elements.\n\nFirst structure (%i elements): %s\n\n"
                         "Second structure (%i elements): %s"
                         % (len_nest1, nest1, len_nest2, nest2))
    _recursive_assert_same_structure(nest1, nest2)


def map_structure(func, *structure):
    if not callable(func):
        raise TypeError("func must be callable!")

    for other in structure[1:]:
        assert_same_structure(structure[0], other)

    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)

    return pack_sequence_as(
        structure[0], [func(*x) for x in entries])


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


class Statistics(object):
    """
    Train/validate loss statistics.
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return safe_exp(self.loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, step, epoch, batch, n_batches, lr, batch_size, best_bleu):
        t = self.elapsed_time()

        out_info = ("Step %d, Epoch %d, %d/%d| lr: %.6f| words: %d| "
                    "acc: %.2f| ppl: %.2f| %.1f tgt tok/s| %.2f s elapsed | best bleu is: ") \
                   % (step, epoch, batch, n_batches, lr, int(batch_size), self.accuracy(), self.ppl(), \
                      self.n_words / (t + 1e-5), time.time() - self.start_time)
        # for i in range(len(best_bleu)):
        #     out_info += '%.2f, ' % best_bleu[i]
        out_info += '%.2f' % best_bleu
        print(out_info)
        sys.stdout.flush()

    def print_valid(self, step):
        t = self.elapsed_time()
        out_info = ("Valid at step %d: acc %.2f, ppl: %.2f, %.1f tgt tok/s, %.2f s elapsed") % \
                   (step, self.accuracy(), self.ppl(), self.n_words / (t + 1e-5),
                    time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()


class Saver(object):
    """ Saver to save and restore objects.

    Saver only accept objects which contain two method: ```state_dict``` and ```load_state_dict```
    """

    def __init__(self, save_prefix, num_max_keeping=1):

        self.save_prefix = save_prefix.rstrip(".")

        save_dir = os.path.dirname(self.save_prefix)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir

        if os.path.exists(self.save_prefix):
            with open(self.save_prefix) as f:
                save_list = f.readlines()
            save_list = [line.strip() for line in save_list]
        else:
            save_list = []

        self.save_list = save_list
        self.num_max_keeping = num_max_keeping

    @staticmethod
    def savable(obj):

        if hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"):
            return True
        else:
            return False

    def save(self, global_step, **kwargs):

        state_dict = dict()

        for key, obj in kwargs.items():
            if self.savable(obj):
                state_dict[key] = obj.state_dict()

        saveto_path = '{0}.{1}'.format(self.save_prefix, global_step)
        torch.save(state_dict, saveto_path)

        self.save_list.append(os.path.basename(saveto_path))

        if len(self.save_list) > self.num_max_keeping:
            out_of_date_state_dict = self.save_list.pop(0)
            os.remove(os.path.join(self.save_dir, out_of_date_state_dict))

        with open(self.save_prefix, "w") as f:
            f.write("\n".join(self.save_list))

    def load_latest(self, **kwargs):

        if len(self.save_list) == 0:
            return

        latest_path = os.path.join(self.save_dir, self.save_list[-1])

        state_dict = torch.load(latest_path)

        for name, obj in kwargs.items():
            if self.savable(obj):

                if name not in state_dict:
                    print("Warning: {0} has no content saved!".format(name))
                else:
                    print("Loading {0}".format(name))
                    obj.load_state_dict(state_dict[name])
