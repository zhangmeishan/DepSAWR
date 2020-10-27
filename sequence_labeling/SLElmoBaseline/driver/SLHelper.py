class SequenceLabeler(object):
    def __init__(self, model, elmo):
        self.model = model
        self.elmo = elmo
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def get_elmo(self, in_words):
        return self.elmo.batch_to_embeddings(in_words)

    def forward(self, words, predicts, masks):

        if self.use_cuda:
            predicts = predicts.cuda(self.device)
            masks = masks.cuda(self.device)

        elmos, _ = self.get_elmo(words)
        label_scores = self.model.forward(elmos, predicts, masks)
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

    def labeler(self, words, predicts, masks):
        if words is not None:
            self.forward(words, predicts, masks)

        predict_labels = self.model.decode(self.label_scores, masks)

        return predict_labels
