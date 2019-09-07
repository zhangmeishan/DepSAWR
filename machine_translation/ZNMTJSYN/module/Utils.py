import torch
import numpy as np
from torch.autograd import Variable
from data.Vocab import NMTVocab
import time
import math
import sys



_FLOAT32_INF = np.float32(np.finfo('float32').max / 10)


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


def tile_batch(x, multiplier, batch_dim=0):
    """
    :type x: Variable
    """
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim+1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled

def mask_scores(scores, beam_mask):

    """
    :type scores: Variable
    :param scores: [B, Bm, N]

    :type beam_mask: Variable
    :param beam_mask: [B, Bm]
    """

    vocab_size = scores.size(-1)

    finished_row = beam_mask.new(vocab_size, ).zero_() + float(_FLOAT32_INF)

    # If beam finished, only PAD could be generated afterwards.
    finished_row[NMTVocab.EOS] = 0.0

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

def reranking_beams(word_ids, scores):

    word_ids = word_ids.cpu().numpy()
    scores = scores.cpu().numpy()

    # Reranking beams
    reranked_beams = np.argsort(scores, axis=1)
    reranked_word_ids = np.ones_like(word_ids) * NMTVocab.PAD

    for b in range(scores.shape[0]):
        for ii in reranked_beams[b]:
            reranked_word_ids[b, ii] = word_ids[b, ii]

    reranked_word_ids = reranked_word_ids.tolist()

    return reranked_word_ids


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


def np_pad_batch_2D(samples, pad, batch_first=True, volatile=False, cuda=True):

    batch_size = len(samples)

    sizes = [len(s) for s in samples]
    max_size = max(sizes)

    x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

    for ii in range(batch_size):
        x_np[ii, :sizes[ii]] = samples[ii]

    if batch_first is False:
        x_np = np.transpose(x_np, [1, 0])

    x = Variable(torch.LongTensor(x_np).contiguous(),
                 volatile=volatile)
    if cuda is True:
        x = x.cuda()

    return x

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

    def print_out(self, step, epoch, batch, n_batches):
        t = self.elapsed_time()
        out_info = ("Step %d, Epoch %d, %d/%d| acc: %.2f| ppl: %.2f| %.1f tgt tok/s| %.2f s elapsed") \
                   % (step, epoch, batch, n_batches,self.accuracy(), self.ppl(), \
                    self.n_words / (t + 1e-5), time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()

    def print_valid(self, step):
        t = self.elapsed_time()
        out_info = ("Valid at step %d: acc %.2f, ppl: %.2f, %.1f tgt tok/s, %.2f s elapsed") % \
              (step, self.accuracy(), self.ppl(), self.n_words / (t + 1e-5),
               time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()