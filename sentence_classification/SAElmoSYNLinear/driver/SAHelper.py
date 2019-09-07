import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SentenceClassifier(object):
    def __init__(self, model, elmo, vocab):
        self.model = model
        self.elmo = elmo
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def get_elmo(self, in_words):
         return self.elmo.batch_to_embeddings(in_words)

    def forward(self, words, actions, word_indexes, masks):
        word_elmos, _ = self.get_elmo(words)
        word_elmos = word_elmos.permute(0, 2, 3, 1).detach()
        batch_size, max_length, elmo_dims, elmo_layers = \
            word_elmos.size()
        mixed_max_length = actions.size(1)
        elmos = Variable(word_elmos.data.new(batch_size, \
                         mixed_max_length, elmo_dims, elmo_layers).zero_(), \
                         requires_grad=False)

        for b in range(batch_size):
            for idx, index in enumerate(word_indexes[b]):
                elmos[b, index] = word_elmos[b, idx]

        if self.use_cuda:
            masks = masks.cuda(self.device)
            actions = actions.cuda(self.device)

        tag_logits = self.model.forward(elmos, actions, masks)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        true_tags = Variable(true_tags, requires_grad=False)
        if self.use_cuda: true_tags = true_tags.cuda()
        loss = F.cross_entropy(self.tag_logits, true_tags)

        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.data.max(1)[1].cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, words, actions, word_indexes, masks):
        if words is not None:
            self.forward(words, actions, word_indexes, masks)
        pred_tags = self.tag_logits.data.max(1)[1].cpu()
        return pred_tags
