from data.DataLoader import *
from module.Utils import *
from model.ParserModel import *


class NMTHelper(object):
    def __init__(self, model, parser, critic, src_vocab, tgt_vocab,
                 dep_vocab, config, parser_config):
        self.model = model
        self.parser = parser
        self.critic = critic

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.dep_vocab = dep_vocab

        self.src_pad = src_vocab.pad()
        self.src_bos = src_vocab.bos()
        self.tgt_pad = tgt_vocab.pad()
        self.tgt_eos = tgt_vocab.eos()
        self.tgt_bos = tgt_vocab.bos()

        self.config = config
        self.parser_config = parser_config

        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        # for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            # idx = int(len(src_input) - 1)
            word_ids, extword_ids, indices = self.dep_input_id(src_input)
            self.train_data[0].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input),
                                       word_ids, extword_ids, indices))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        # for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def src_data_id(self, src_input):
        result = [self.src_vocab.word2id(cur_word) for cur_word in src_input]

        return [self.src_bos] + result

    def tgt_data_id(self, tgt_input):
        result = [self.tgt_vocab.word2id(cur_word) for cur_word in tgt_input]

        return [self.tgt_bos] + result + [self.tgt_eos]

    def dep_input_id(self, src_input):
        lower_src_input, indices = [], [0]
        full_word, cur_idx = "", 1
        for cur_word in src_input:
            sub_word = cur_word.lower()
            indices.append(cur_idx)
            if sub_word.endswith("@@"):
                full_word = full_word + sub_word[0:-2]
            else:
                lower_src_input.append(full_word + sub_word)
                cur_idx = cur_idx + 1
                full_word = ""

        word_ids = [self.dep_vocab.ROOT] + self.dep_vocab.word2id(lower_src_input)
        extword_ids = [self.dep_vocab.ROOT] + self.dep_vocab.extword2id(lower_src_input)
        return word_ids, extword_ids, indices

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            word_ids, extword_ids, indices = self.dep_input_id(src_input)
            eval_data.append((self.src_data_id(src_input), src_input,
                              word_ids, extword_ids, indices))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        dep_lengths = [len(batch[i][2]) for i in range(batch_size)]
        max_dep_length = int(np.max(dep_lengths))

        tgt_lengths = [len(batch[i][1]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)
        tgt_words = torch.zeros([batch_size, max_tgt_length], dtype=torch.int64, requires_grad=False)

        src_words = src_words.fill_(self.src_pad)
        tgt_words = tgt_words.fill_(self.tgt_pad)

        dep_words = torch.zeros([batch_size, max_dep_length], dtype=torch.int64, requires_grad=False)
        dep_extwords = torch.zeros([batch_size, max_dep_length], dtype=torch.int64, requires_grad=False)
        dep_masks = torch.zeros([batch_size, max_dep_length], dtype=torch.float, requires_grad=False)

        indices = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)

        dep_words = dep_words.fill_(self.dep_vocab.PAD)
        dep_extwords = dep_extwords.fill_(self.dep_vocab.PAD)
        indices = indices.fill_(max_src_length-1)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, word in enumerate(instance[1]):
                tgt_words[b, index] = word
            for index, word in enumerate(instance[2]):
                dep_words[b, index] = word
                dep_masks[b, index] = 1.0
            for index, word in enumerate(instance[3]):
                dep_extwords[b, index] = word
            for index, sid in enumerate(instance[4]):
                indices[b, index] = sid

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)
            dep_words = dep_words.cuda(self.device)
            dep_masks = dep_masks.cuda(self.device)
            dep_extwords = dep_extwords.cuda(self.device)
            indices = indices.cuda(self.device)

        return src_words, tgt_words, src_lengths, tgt_lengths, \
               dep_words, dep_extwords, dep_masks, indices

    def source_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        dep_lengths = [len(batch[i][2]) for i in range(batch_size)]
        max_dep_length = int(np.max(dep_lengths))

        src_words = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)
        src_words = src_words.fill_(self.src_pad)

        dep_words = torch.zeros([batch_size, max_dep_length], dtype=torch.int64, requires_grad=False)
        dep_extwords = torch.zeros([batch_size, max_dep_length], dtype=torch.int64, requires_grad=False)
        dep_masks = torch.zeros([batch_size, max_dep_length], dtype=torch.float, requires_grad=False)

        indices = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)

        dep_words = dep_words.fill_(self.dep_vocab.PAD)
        dep_extwords = dep_extwords.fill_(self.dep_vocab.PAD)
        indices = indices.fill_(max_src_length-1)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, word in enumerate(instance[2]):
                dep_words[b, index] = word
                dep_masks[b, index] = 1.0
            for index, word in enumerate(instance[3]):
                dep_extwords[b, index] = word
            for index, sid in enumerate(instance[4]):
                indices[b, index] = sid

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            dep_words = dep_words.cuda(self.device)
            dep_masks = dep_masks.cuda(self.device)
            dep_extwords = dep_extwords.cuda(self.device)
            indices = indices.cuda(self.device)

        return src_words, src_lengths, dep_words, dep_extwords, dep_masks, indices

    def parse_one_batch(self, dep_words, dep_extwords, dep_masks, indices, bTrain):
        if bTrain and self.config.parser_tune == 1:
            self.parser.train()
        else:
            self.parser.eval()

        parser_outputs = self.parser.forward(dep_words, dep_extwords, dep_masks)
        # move the hidden vector of the first fake word to the last position
        proof_outputs = []

        for parser_output in parser_outputs:
            layer_indices = indices.unsqueeze(indices, dim=2).expand(-1, -1, parser_output.size(2))
            proof_output = torch.gather(parser_output, 1, layer_indices)
            proof_outputs.append(proof_output)

        return proof_outputs

    def compute_forward(self, seqs_x, seqs_y, xlengths,
                        dep_words, dep_extwords, dep_masks, indices,
                        normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """

        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, indices, True)

        with torch.enable_grad():
            logits = self.model(seqs_x, synx, y_inp, lengths=xlengths)

            loss = self.critic(inputs=logits,
                               labels=y_label,
                               normalization=normalization)

            loss = loss.sum()
        torch.autograd.backward(loss)

        mask = y_label.detach().ne(self.tgt_pad)
        pred = logits.detach().max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.detach().eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().float()

        stats = Statistics(loss.item(), num_total, num_correct)

        return loss, stats

    def train_one_batch(self, batch):
        self.model.train()
        # self.model.zero_grad()
        src_words, tgt_words, src_lengths, tgt_lengths, \
        dep_words, dep_extwords, dep_masks, indices = self.pair_data_variable(batch)

        loss, stat = self.compute_forward(src_words, tgt_words, src_lengths,
                                          dep_words, dep_extwords, dep_masks, indices)

        return stat

    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            src_words, src_lengths, dep_words, \
            dep_extwords, dep_masks, indices = self.source_data_variable(batch)

            allHyp = self.translate_batch(src_words, src_lengths,
                                          dep_words, dep_extwords, dep_masks, indices)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            # for idx in range(batch_size):
            #     if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
            #         all_hyp_inds[idx].pop()

            all_hyp_words = []
            for idxs in all_hyp_inds:
                all_hyp_words += [[self.tgt_vocab.id2word(idx) for idx in idxs]]

            for idx, instance in enumerate(batch):
                result['\t'.join(instance[1])] = all_hyp_words[idx]

        return result

    def translate_batch(self, src_inputs, src_input_lengths,
                        dep_words, dep_extwords, dep_masks, indices):

        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, indices, False)

        word_ids = self.model(src_inputs, synx, lengths=src_input_lengths,
                              mode="infer", beam_size=self.config.beam_size)
        # print(word_ids.size())

        word_ids = word_ids.cpu().numpy().tolist()
        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if (wid != self.tgt_eos and wid != self.tgt_pad)] for line in sent_t]
            result.append(sent_t)

        return result
