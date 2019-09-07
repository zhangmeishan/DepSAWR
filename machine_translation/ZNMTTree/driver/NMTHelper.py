import torch.nn.functional as F
from torch.autograd import Variable
from data.DataLoader import *
from module.Utils import *
from data.Vocab import *
from module.Tree import *


class NMTHelper(object):
    def __init__(self, model, critic, src_vocab, tgt_vocab, config):
        self.model = model
        self.critic = critic
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        #for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            cur_length = len(src_input)
            words, heads, rels = self.src_data_id(src_input)
            root, tree = creatTree(heads)
            if root.depth() > cur_length:
                forms = [the_form + "_" + str(the_head) + "_" +  the_rel for the_form, the_head, the_rel in src_input]
                print("strange: " + '_'.join(forms))
            self.train_data[0].append((words, rels, heads, self.tgt_data_id(tgt_input)))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        #for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def src_data_id(self, src_input):
        words, heads, rels = [SRCVocab.ROOT], [-1], [SRCVocab.ROOT]
        for unit in src_input:
            words.append(self.src_vocab.word2id(unit[0]))
            rels.append(self.src_vocab.rel2id(unit[2]))
            heads.append(unit[1])
        return words, heads, rels

    def tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.word2id(tgt_input)
        return [TGTVocab.BOS] + result + [TGTVocab.EOS]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            words, heads, rels = self.src_data_id(src_input)
            forms = [the_form for the_form, the_head, the_rel in src_input]
            eval_data.append((words, rels, heads, forms))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        tgt_lengths = [len(batch[i][3]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(SRCVocab.PAD), requires_grad=False)
        src_rels = Variable(torch.LongTensor(batch_size, max_src_length).fill_(SRCVocab.PAD), requires_grad=False)
        tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(TGTVocab.PAD), requires_grad=False)
        heads = []

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, rel in enumerate(instance[1]):
                src_rels[b, index] = rel
            heads.append(instance[2])
            for index, word in enumerate(instance[3]):
                tgt_words[b, index] = word
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            src_rels = src_rels.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)

        return src_words, src_rels, heads, src_lengths, tgt_words

    def source_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(SRCVocab.PAD), requires_grad=False)
        src_rels = Variable(torch.LongTensor(batch_size, max_src_length).fill_(SRCVocab.PAD), requires_grad=False)
        heads = []

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, rel in enumerate(instance[1]):
                src_rels[b, index] = rel
            heads.append(instance[2])
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            src_rels = src_rels.cuda(self.device)

        return src_words, src_rels, heads, src_lengths

    def compute_forward(self, seqs_x, seqs_rel, seqs_head, seqs_y, xlengths, normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """

        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        dec_outs = self.model(seqs_x, seqs_rel, seqs_head, y_inp, lengths=xlengths)

        loss = self.critic(generator=self.model.generator,
                      normalization=normalization,
                      dec_outs=dec_outs,
                      labels=y_label)

        mask = y_label.data.ne(TGTVocab.PAD)
        pred = self.model.generator(dec_outs).data.max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.data.eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().float()

        stats = Statistics(loss.item(), num_total, num_correct)

        return loss, stats


    def train_one_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        src_words, src_rels, heads, src_lengths, tgt_words = self.pair_data_variable(batch)
        loss, stat = self.compute_forward(src_words, src_rels, heads, tgt_words, src_lengths)
        loss = loss / self.config.update_every
        loss.backward()

        return stat


    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_rels, heads, src_lengths = self.source_data_variable(batch)
            allHyp = self.translate_batch(src_words, src_rels, heads, src_lengths)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            for idx in range(batch_size):
                if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
                    all_hyp_inds[idx].pop()
            all_hyp_words = [self.tgt_vocab.id2word(idxs) for idxs in all_hyp_inds]
            for idx, instance in enumerate(batch):
                result['\t'.join(instance[-1])] = all_hyp_words[idx]

        return result


    def translate_batch(self, src_inputs, src_rels, src_heads, src_input_lengths):
        word_ids = self.model(src_inputs, src_rels, src_heads, lengths=src_input_lengths, \
                              mode="infer", beam_size=self.config.beam_size)
        word_ids = word_ids.cpu().numpy().tolist()

        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != TGTVocab.PAD] for line in sent_t]
            result.append(sent_t)

        return result
