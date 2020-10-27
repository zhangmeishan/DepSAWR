import torch


class SequenceLabeler(object):
    def __init__(self, model, bert):
        self.model = model
        self.bert = bert
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def dump_bert(self, bert_indices, bert_segments, bert_pieces, bTrain):
        if self.model.config.bert_tune == 1 and bTrain:
            bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)
            return bert_outputs
        else:
            with torch.no_grad():
                bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)
                return bert_outputs

    def forward(self, inputs, actions, predicts, masks, word_indexes, indices):
        bert_indices, bert_segments, bert_pieces = inputs[0], inputs[1], inputs[2]
        if self.use_cuda:
            bert_indices = bert_indices.cuda(self.device)
            bert_segments = bert_segments.cuda(self.device)
            bert_pieces = bert_pieces.cuda(self.device)
            predicts = predicts.cuda(self.device)
            masks = masks.cuda(self.device)
            actions = actions.cuda(self.device)
            indices = indices.cuda(self.device)

        bert_outputs = self.dump_bert(bert_indices, bert_segments, bert_pieces, self.model.training)
        rbert_outputs = []

        for bert_output in bert_outputs:
            batch_size, max_length, bert_dims = bert_output.size()
            mixed_max_length = actions.size(1)
            rbert_output = bert_output.data.new(batch_size, mixed_max_length, bert_dims).zero_()

            for b in range(batch_size):
                for idx, index in enumerate(word_indexes[b]):
                    rbert_output[b, index] = bert_output[b, idx]

            rbert_outputs.append(rbert_output)

        label_scores = self.model.forward(rbert_outputs, actions, predicts, masks, indices)
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

    def labeler(self, inputs, actions, predicts, masks, word_indexes, indices, wmasks):
        if inputs is not None:
            self.forward(inputs, actions, predicts, masks, word_indexes, indices)

        predict_labels = self.model.decode(self.label_scores, wmasks)

        return predict_labels
