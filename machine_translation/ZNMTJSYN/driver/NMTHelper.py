import torch.nn.functional as F
from torch.autograd import Variable
from data.DataLoader import *
from module.Utils import *
from data.Vocab import NMTVocab
from model.ParserModel import *


class NMTHelper(object):
    def __init__(self, model, parser, critic, src_vocab, tgt_vocab, dep_vocab, config, parser_config):
        self.model = model
        self.parser = parser
        self.critic = critic
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.dep_vocab = dep_vocab
        self.config = config
        self.parser_config = parser_config

    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        #for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            #idx = int(len(src_input) - 1)
            word_ids, extword_ids = self.dep_input_id(src_input)
            self.train_data[0].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input), \
                                       word_ids, extword_ids))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        #for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_valid_data(self, src_inputs, tgt_inputs):
        self.valid_data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            word_ids, extword_ids = self.dep_input_id(src_input)
            self.valid_data.append((self.src_data_id(src_input), self.tgt_data_id(tgt_input), \
                                    word_ids, extword_ids))
        self.valid_size = len(self.valid_data)

    def src_data_id(self, src_input):
        result = self.src_vocab.word2id(src_input)
        return result + [self.src_vocab.EOS]

    def tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.word2id(tgt_input)
        return [self.tgt_vocab.BOS] + result + [self.tgt_vocab.EOS]

    def dep_input_id(self, src_input):
        lower_src_input = [cur_word.lower() for cur_word in src_input]
        word_ids = [self.dep_vocab.ROOT] + self.dep_vocab.word2id(lower_src_input)
        extword_ids = [self.dep_vocab.ROOT] + self.dep_vocab.extword2id(lower_src_input)
        return word_ids, extword_ids

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            word_ids, extword_ids = self.dep_input_id(src_input)
            eval_data.append((self.src_data_id(src_input), src_input, word_ids, extword_ids))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        tgt_lengths = [len(batch[i][1]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
        tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(NMTVocab.PAD), requires_grad=False)

        dep_words = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_extwords = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_masks = Variable(torch.Tensor(batch_size, max_src_length).zero_(), requires_grad=False)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, word in enumerate(instance[1]):
                tgt_words[b, index] = word

            for index, word in enumerate(instance[2]):
                dep_words[b, index] = word
                dep_masks[b, index] = 1
            for index, word in enumerate(instance[3]):
                dep_extwords[b, index] = word

            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)
            dep_words = dep_words.cuda(self.device)
            dep_extwords = dep_extwords.cuda(self.device)
            dep_masks = dep_masks.cuda(self.device)

        return src_words, tgt_words, src_lengths, tgt_lengths, dep_words, dep_extwords, dep_masks

    def source_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)

        dep_words = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_extwords = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_masks = Variable(torch.Tensor(batch_size, max_src_length).zero_(), requires_grad=False)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word

            for index, word in enumerate(instance[2]):
                dep_words[b, index] = word
                dep_masks[b, index] = 1
            for index, word in enumerate(instance[3]):
                dep_extwords[b, index] = word

            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            dep_words = dep_words.cuda(self.device)
            dep_extwords = dep_extwords.cuda(self.device)
            dep_masks = dep_masks.cuda(self.device)

        return src_words, src_lengths, dep_words, dep_extwords, dep_masks

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
            proof_output = torch.cat(chunks[1:]+chunks[0:1], 0)
            proof_outputs.append(proof_output.transpose(1, 0))

        return proof_outputs

    def compute_forward(self, seqs_x, dep_words, dep_extwords, dep_masks, seqs_y, xlengths, normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """

        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, True)

        dec_outs = self.model(seqs_x, synx, y_inp, lengths=xlengths)

        loss = self.critic(generator=self.model.generator,
                      normalization=normalization,
                      dec_outs=dec_outs,
                      labels=y_label)

        mask = y_label.data.ne(NMTVocab.PAD)
        pred = self.model.generator(dec_outs).data.max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.data.eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().float()

        stats = Statistics(loss.item(), num_total, num_correct)

        return loss, stats

    def train_one_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        src_words, tgt_words, src_lengths, tgt_lengths, dep_words, dep_extwords, dep_masks \
            = self.pair_data_variable(batch)
        loss, stat = self.compute_forward(src_words, dep_words, dep_extwords, dep_masks, tgt_words, src_lengths)
        loss = loss / self.config.update_every
        loss.backward()

        return stat

    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_lengths, dep_words, dep_extwords, dep_masks = self.source_data_variable(batch)
            allHyp = self.translate_batch(src_words, src_lengths, dep_words, dep_extwords, dep_masks)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            for idx in range(batch_size):
                if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
                    all_hyp_inds[idx].pop()
            all_hyp_words = [self.tgt_vocab.id2word(idxs) for idxs in all_hyp_inds]
            for idx, instance in enumerate(batch):
                result['\t'.join(instance[1])] = all_hyp_words[idx]

        return result

    def translate_batch(self, src_inputs, src_input_lengths, dep_words, dep_extwords, dep_masks):
        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, False)

        word_ids = self.model(src_inputs, synx, lengths=src_input_lengths, mode="infer", beam_size=self.config.beam_size)
        word_ids = word_ids.cpu().numpy().tolist()

        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != NMTVocab.PAD] for line in sent_t]
            result.append(sent_t)

        return result
