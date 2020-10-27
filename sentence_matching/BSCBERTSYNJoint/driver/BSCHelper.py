import torch
import torch.nn as nn


class BiSententClassifier(object):
    def __init__(self, model, bert, vocab, parser):
        self.model = model
        self.vocab = vocab
        self.bert = bert
        self.parser = parser
        self.parser_tune = model.config.parser_tune
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.criterion = nn.CrossEntropyLoss()

    def dump_bert(self, bert_indices, bert_segments, bert_pieces, bTrain):
        if self.bert.config.bert_tune == 1 and bTrain:
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

    def forward(self, tinputs, dep_tinputs):
        src_bert_indices, src_bert_segments, src_bert_pieces,\
        tgt_bert_indices, tgt_bert_segments, tgt_bert_pieces,\
        src_lens, src_masks, tgt_lens, tgt_masks = tinputs

        src_dep_words, src_dep_extwords, src_dep_masks, \
        tgt_dep_words, tgt_dep_extwords, tgt_dep_masks = dep_tinputs

        src_bert_outputs = self.dump_bert(src_bert_indices, src_bert_segments,
                                          src_bert_pieces, self.model.training)
        tgt_bert_outputs = self.dump_bert(tgt_bert_indices, tgt_bert_segments,
                                          tgt_bert_pieces, self.model.training)

        src_synx = self.parse_one_batch(src_dep_words, src_dep_extwords, src_dep_masks, self.model.training)
        tgt_synx = self.parse_one_batch(tgt_dep_words, tgt_dep_extwords, tgt_dep_masks, self.model.training)

        new_inputs = (src_bert_outputs, src_synx, src_lens, src_masks, \
                      tgt_bert_outputs, tgt_synx, tgt_lens, tgt_masks)

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

    def classifier(self, tinputs, dep_tinputs):
        if tinputs[0] is not None:
            self.forward(tinputs, dep_tinputs)
        pred_tags = self.tag_logits.detach().max(1)[1].cpu()
        return pred_tags
