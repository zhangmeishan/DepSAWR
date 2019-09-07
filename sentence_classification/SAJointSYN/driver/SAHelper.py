import torch.nn.functional as F
from torch.autograd import Variable
import torch

class SentenceClassifier(object):
    def __init__(self, config, model, vocab, parser, dep_vocab):
        self.model = model
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.parser = parser
        self.dep_vocab = dep_vocab
        self.config = config

    def parse_one_batch(self, dep_words, dep_extwords, dep_masks, bTrain):
        if bTrain and self.config.parser_tune == 1:
            self.parser.train()
        else:
            self.parser.eval()

        parser_outputs = self.parser.forward(dep_words, dep_extwords, dep_masks)
        # move the hidden vector of the first fake word to the last position
        proof_outputs = []
        for parser_output in parser_outputs:
            chunks = torch.split(parser_output.transpose(1, 0), split_size_or_sections=1, dim=0)
            proof_output = torch.cat(chunks[1:], 0)
            proof_outputs.append(proof_output.transpose(1, 0))

        return proof_outputs

    def forward(self, words, extwords, masks, dep_words, dep_extwords, dep_masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device)
            dep_words, dep_extwords = dep_words.cuda(self.device), dep_extwords.cuda(self.device)
            masks, dep_masks = masks.cuda(self.device), dep_masks.cuda(self.device)

        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, self.model.training)

        tag_logits = self.model.forward(words, extwords, synx, masks)
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

    def classifier(self, words, extwords, masks, dep_words, dep_extwords, dep_masks):
        if words is not None:
            self.forward(words, extwords, masks, dep_words, dep_extwords, dep_masks)

        pred_tags = self.tag_logits.data.max(1)[1].cpu()

        return pred_tags
