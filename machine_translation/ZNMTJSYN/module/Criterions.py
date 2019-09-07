import torch
import torch.nn as nn
from torch.autograd import Variable
from data.Vocab import NMTVocab

class Critierion(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self):
        super(Critierion, self).__init__()

    def _compute_loss(self, generator, *args, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        raise NotImplementedError

    def forward(self, generator, normalization=1.0, **kwargs):
        return self._compute_loss(generator, **kwargs).div(normalization)


class NMTCritierion(Critierion):
    """
    TODO:
    1. Add label smoothing
    """
    def __init__(self, padding_idx=NMTVocab.PAD, label_smoothing=0.0):

        super().__init__()
        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=NMTVocab.PAD)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, generator, dec_outs, labels):

        scores = generator(self._bottle(dec_outs)) # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            tdata = gtruth.data

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))

            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()

            tmp_ = one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)

        loss = self.criterion(scores, gtruth)

        return loss
