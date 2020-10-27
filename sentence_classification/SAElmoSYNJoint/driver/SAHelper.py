import torch
import torch.nn.functional as F


class SentenceClassifier(object):
    def __init__(self, model, elmo, vocab, parser, dep_vocab):
        self.model = model
        self.elmo = elmo
        self.vocab = vocab
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.parser = parser
        self.dep_vocab = dep_vocab

    def get_elmo(self, in_words):
         return self.elmo.batch_to_embeddings(in_words)

    def parse_one_batch(self, dep_words, dep_extwords, dep_masks, bTrain):
        if bTrain and self.model.config.parser_tune == 1:
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

    def forward(self, words, masks, dep_words, dep_extwords, dep_masks):
        if self.use_cuda:
            dep_words, dep_extwords = dep_words.cuda(self.device), dep_extwords.cuda(self.device)
            masks, dep_masks = masks.cuda(self.device), dep_masks.cuda(self.device)

        elmos, _ = self.get_elmo(words)
        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, self.model.training)

        tag_logits = self.model.forward(elmos, synx, masks)
        # cache
        self.tag_logits = tag_logits

    def compute_loss(self, true_tags):
        if self.use_cuda: true_tags = true_tags.cuda()
        loss = F.cross_entropy(self.tag_logits, true_tags)

        return loss

    def compute_accuracy(self, true_tags):
        b, l = self.tag_logits.size()
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        tag_correct = pred_tags.eq(true_tags).cpu().sum()

        return tag_correct, b

    def classifier(self, words, masks, dep_words, dep_extwords, dep_masks):
        if words is not None:
            self.forward(words, masks, dep_words, dep_extwords, dep_masks)

        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
