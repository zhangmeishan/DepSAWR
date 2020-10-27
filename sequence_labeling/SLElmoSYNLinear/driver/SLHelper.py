class SequenceLabeler(object):
    def __init__(self, model, elmo):
        self.model = model
        self.elmo = elmo
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def get_elmo(self, in_words):
        return self.elmo.batch_to_embeddings(in_words)

    def forward(self, words, actions, predicts, masks, word_indexes, indices):
        word_elmos, _ = self.get_elmo(words)
        word_elmos = word_elmos.permute(0, 2, 3, 1).detach()
        batch_size, max_length, elmo_dims, elmo_layers = \
            word_elmos.size()
        mixed_max_length = actions.size(1)
        elmos = word_elmos.data.new(batch_size, mixed_max_length, elmo_dims, elmo_layers).zero_()

        for b in range(batch_size):
            for idx, index in enumerate(word_indexes[b]):
                elmos[b, index] = word_elmos[b, idx]

        if self.use_cuda:
            predicts = predicts.cuda(self.device)
            masks = masks.cuda(self.device)
            actions = actions.cuda(self.device)
            indices = indices.cuda(self.device)

        label_scores = self.model.forward(elmos, actions, predicts, masks, indices)
        # cache
        self.label_scores = label_scores

    def compute_loss(self, answers, wmasks):
        if self.use_cuda:
            answers = answers.cuda(self.device)
            wmasks = wmasks.cuda(self.device)
        loss = self.model.compute_loss(self.label_scores, answers, wmasks)

        return loss

    def compute_accuracy(self, answers):
        scores = self.label_scores.data
        target = answers.data
        pred = scores.max(2)[1]
        non_padding = target.ne(self.model.PAD)
        num_words = non_padding.sum()
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return num_correct, num_words

    def labeler(self, words, actions, predicts, masks, word_indexes, indices, wmasks):
        if words is not None:
            self.forward(words, actions, predicts, masks, word_indexes, indices)

        predict_labels = self.model.decode(self.label_scores, wmasks)

        return predict_labels
