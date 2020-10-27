import torch.nn.functional as F


class SentenceClassifier(object):
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, words, extwords, rels, heads, masks, lengths):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device)
            rels = rels.cuda(self.device)
            masks = masks.cuda(self.device)

        tag_logits = self.model.forward(words, extwords, rels, heads, masks, lengths)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        if self.use_cuda: true_tags = true_tags.cuda()
        loss = F.cross_entropy(self.tag_logits, true_tags)

        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.data.max(1)[1].cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, words, extwords, rels, heads, masks, lengths):
        if words is not None:
            self.forward(words, extwords, rels, heads, masks, lengths)
        pred_tags = self.tag_logits.data.max(1)[1].cpu()
        return pred_tags
