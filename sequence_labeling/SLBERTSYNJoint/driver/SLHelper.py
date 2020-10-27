import torch


class SequenceLabeler(object):
    def __init__(self, model, bert, parser):
        self.model = model
        self.bert = bert
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.parser = parser
        self.parser_tune = model.config.parser_tune

    def dump_bert(self, bert_indices, bert_segments, bert_pieces, bTrain):
        if self.model.config.bert_tune == 1 and bTrain:
            bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)
            return bert_outputs
        else:
            with torch.no_grad():
                bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)
                return bert_outputs

    def parse_one_batch(self, dep_words, dep_extwords, dep_masks, bTrain):
        if bTrain and self.parser_tune == 1:
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

    def forward(self, inputs, predicts, masks, dep_words, dep_extwords, dep_masks):
        bert_indices, bert_segments, bert_pieces = inputs[0], inputs[1], inputs[2]
        if self.use_cuda:
            bert_indices = bert_indices.cuda(self.device)
            bert_segments = bert_segments.cuda(self.device)
            bert_pieces = bert_pieces.cuda(self.device)
            predicts = predicts.cuda(self.device)
            dep_words, dep_extwords = dep_words.cuda(self.device), dep_extwords.cuda(self.device)
            masks, dep_masks = masks.cuda(self.device), dep_masks.cuda(self.device)

        bert_outputs = self.dump_bert(bert_indices, bert_segments, bert_pieces, self.model.training)
        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, self.model.training)

        label_scores = self.model.forward(bert_outputs, predicts, synx, masks)
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

    def labeler(self, inputs, predicts, masks, dep_words, dep_extwords, dep_masks):
        if inputs is not None:
            self.forward(inputs, predicts, masks, dep_words, dep_extwords, dep_masks)

        predict_labels = self.model.decode(self.label_scores, masks)

        return predict_labels
