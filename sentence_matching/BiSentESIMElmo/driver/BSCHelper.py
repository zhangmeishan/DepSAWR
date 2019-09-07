import torch.nn.functional as F
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

    def get_elmo(self, in_words):
         return self.elmo.batch_to_embeddings(in_words)

    def forward(self, tinputs, src_forms, tgt_forms):
        src_words, src_lens, src_masks, \
        tgt_words, tgt_lens, tgt_masks = tinputs

        src_elmos, _ = self.get_elmo(src_forms)
        tgt_elmos, _ = self.get_elmo(tgt_forms)

        new_inputs = (src_words, src_elmos, src_lens, src_masks, \
                      tgt_words, tgt_elmos, tgt_lens, tgt_masks)

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

    def classifier(self, tinputs, src_words, tgt_words):
        if tinputs[0] is not None:
            self.forward(tinputs, src_words, tgt_words)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
