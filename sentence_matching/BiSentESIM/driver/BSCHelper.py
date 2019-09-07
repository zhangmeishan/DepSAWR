import torch.nn.functional as F
import torch.nn as nn


class BiSententClassifier(object):
    def __init__(self, model, extword_embed, vocab):
        self.model = model
        self.vocab = vocab
        self.extword_embed = extword_embed
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, tinputs):
        src_words, src_extwords, src_lens, src_masks, \
        tgt_words, tgt_extwords, tgt_lens, tgt_masks = tinputs

        src_ext_embs = self.extword_embed(src_extwords)
        tgt_ext_embs = self.extword_embed(tgt_extwords)

        new_inputs = (src_words, src_ext_embs, src_lens, src_masks, \
                      tgt_words, tgt_ext_embs, tgt_lens, tgt_masks)

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

    def classifier(self, tinputs):
        if tinputs[0] is not None:
            self.forward(tinputs)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
