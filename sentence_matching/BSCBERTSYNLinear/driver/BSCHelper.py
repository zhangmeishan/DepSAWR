import torch
import torch.nn as nn


class BiSententClassifier(object):
    def __init__(self, model, elmo, vocab):
        self.model = model
        self.vocab = vocab
        self.elmo = elmo
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()

    def padded_bert(self, bert_indices, bert_segments, bert_pieces,
                    word_indexes, mixed_max_length, bTrain):
        if self.bert.config.bert_tune == 1 and bTrain:
            bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)
        else:
            with torch.no_grad():
                bert_outputs = self.bert(bert_indices, bert_segments, bert_pieces)

        refined_bert_outputs = []

        for bert_output in bert_outputs:
            batch_size, max_length, bert_dims = bert_outputs[0].size()
            refined_bert_output = bert_output.data.new(batch_size, mixed_max_length, bert_dims).zero_()

            for b in range(batch_size):
                for idx, index in enumerate(word_indexes[b]):
                    refined_bert_output[b, index] = refined_bert_output[b, idx]

            refined_bert_outputs.append(refined_bert_output)

        return refined_bert_outputs

    def forward(self, tinputs, src_indexes, tgt_indexes):
        src_bert_indices, src_bert_segments, src_bert_pieces, \
        src_actions, src_lens, src_masks, \
        tgt_bert_indices, tgt_bert_segments, tgt_bert_pieces, \
        tgt_actions, tgt_lens, tgt_masks = tinputs

        src_mixed_length = src_actions.size(1)
        tgt_mixed_length = tgt_actions.size(1)
        src_berts = self.padded_bert(src_bert_indices, src_bert_segments, src_bert_pieces,
                                    src_indexes, src_mixed_length)
        tgt_berts = self.padded_bert(tgt_bert_indices, tgt_bert_segments, tgt_bert_pieces,
                                    tgt_indexes, tgt_mixed_length)

        new_inputs = (src_berts, src_actions, src_lens, src_masks, \
                      tgt_berts, tgt_actions, tgt_lens, tgt_masks)

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

    def classifier(self, tinputs, src_indexes, tgt_indexes):
        if tinputs[0] is not None:
            self.forward(tinputs, src_indexes, tgt_indexes)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
