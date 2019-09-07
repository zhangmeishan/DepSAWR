from torch.autograd import Variable
import torch.nn as nn


class BiSententClassifier(object):
    def __init__(self, model, elmo, vocab):
        self.model = model
        self.vocab = vocab
        self.elmo = elmo
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()

    def padded_elmos(self, in_words, in_indexes, mixed_max_length):
        word_elmos, _ = self.elmo.batch_to_embeddings(in_words)
        word_elmos = word_elmos.permute(0, 2, 3, 1).detach()
        batch_size, max_length, elmo_dims, elmo_layers = word_elmos.size()
        elmos = Variable(word_elmos.data.new(batch_size, \
                         mixed_max_length, elmo_dims, elmo_layers).zero_(), \
                         requires_grad=False)

        for b in range(batch_size):
            for idx, index in enumerate(in_indexes[b]):
                elmos[b, index] = word_elmos[b, idx]

        return elmos


    def forward(self, tinputs, src_forms, src_indexes, tgt_forms, tgt_indexes):
        src_words, src_actions, src_lens, src_masks, \
        tgt_words, tgt_actions, tgt_lens, tgt_masks = tinputs

        src_max_length = src_actions.size(1)
        tgt_max_length = tgt_actions.size(1)
        src_elmos = self.padded_elmos(src_forms, src_indexes, src_max_length)
        tgt_elmos = self.padded_elmos(tgt_forms, tgt_indexes, tgt_max_length)

        new_inputs = (src_words, src_actions, src_elmos, src_lens, src_masks, \
                      tgt_words, tgt_actions, tgt_elmos, tgt_lens, tgt_masks)

        tag_logits = self.model(new_inputs)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        loss = self.criterion(self.tag_logits, true_tags)
        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        true_tags = true_tags.detach().cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, tinputs, src_forms, src_indexes, tgt_forms, tgt_indexes):
        if tinputs[0] is not None:
            self.forward(tinputs, src_forms, src_indexes, tgt_forms, tgt_indexes)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
