class SequenceLabeler(object):
    def __init__(self, model):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def forward(self, words, extwords, predicts, masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device)
            predicts = predicts.cuda(self.device)
            masks = masks.cuda(self.device)

        label_scores = self.model.forward(words, extwords, predicts, masks)
        # cache
        self.label_scores = label_scores

    def compute_loss(self, answers, masks):
        if self.use_cuda:
            answers = answers.cuda(self.device)
            masks = masks.cuda(self.device)
        loss = self.model.compute_loss(self.label_scores, answers, masks)

        return loss

    def compute_accuracy(self, answers):
        scores = self.label_scores.data
        target = answers.data
        pred = scores.max(2)[1]
        non_padding = target.ne(self.model.PAD)
        num_words = non_padding.sum()
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return num_correct, num_words

    def labeler(self, words, extwords, predicts, masks):
        if words is not None:
            self.forward(words, extwords, predicts, masks)

        predict_labels = self.model.decode(self.label_scores, masks)

        return predict_labels
