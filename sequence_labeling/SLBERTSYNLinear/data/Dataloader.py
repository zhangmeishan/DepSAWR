from collections import Counter
from data.Vocab import *
from data.Instance import *
import numpy as np
import torch


def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readInstance(infile):
            data.append(sentence)
    return data


def creat_vocab(corpusFile, bert_vocab_file, min_occur_count):
    word_counter = Counter()
    label_counter = Counter()
    with open(corpusFile, 'r') as infile:
        for sentence in readInstance(infile):
            index = 0
            for token in sentence.words:
                if token.wtype == 0: word_counter[token.form] += 1
                if (index < sentence.key_start or index > sentence.key_end) \
                        and token.wtype == 1:
                    label_counter[token.label] += 1
                index = index + 1

    return SLVocab(word_counter, label_counter, bert_vocab_file, min_occur_count)


def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)


def inst2id(inst, vocab):
    inputs, labels = [], []
    for curword in inst.words:
        wordid = vocab.word2id(curword.form) if curword.wtype == 0 else vocab.UNK
        inputs.append(wordid)

    index = 0
    for curlabel in inst.labels:
        if index < inst.wkey_start or index > inst.wkey_end:
            labelid = vocab.label2id(curlabel)
        else:
            labelid = vocab.PAD
        index = index + 1
        inputs.append(labelid)

    return inputs, labels, inst.key_start, inst.key_end, inst


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield insts


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  insts in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    batch_size = len(batch)
    lengths = [batch[b].length for b in range(batch_size)]
    wlengths = [batch[b].wlength for b in range(batch_size)]
    max_length,  max_wlength = 0, 0
    for b in range(0, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]
        if lengths[b] > max_wlength: max_wlength = wlengths[b]

    actions = torch.zeros([batch_size, max_length], dtype=torch.int64, requires_grad=False)
    predicts = torch.zeros([batch_size, max_length], dtype=torch.int64, requires_grad=False)
    masks = torch.zeros([batch_size, max_length], dtype=torch.float, requires_grad=False)
    wmasks = torch.zeros([batch_size, max_wlength], dtype=torch.float, requires_grad=False)
    labels = torch.zeros([batch_size, max_wlength], dtype=torch.int64, requires_grad=False)
    indices = torch.zeros([batch_size, max_wlength], dtype=torch.int64, requires_grad=False)
    indices.fill_(max_wlength-1)
    word_indexes = []
    bert_lengths = []

    bert_token_indices, bert_segments_ids, bert_piece_ids = [], [], []
    b, max_bert_length = 0, 0
    for inputs, wlabels, key_start, key_end, inst in insts_numberize(batch, vocab):
        bert_indice, segments_id, piece_id = vocab.bert_ids(inst.sentence)

        cur_length = len(bert_indice)
        bert_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length
        bert_token_indices.append(bert_indice)
        bert_segments_ids.append(segments_id)
        bert_piece_ids.append(piece_id)

        for index in range(inst.length):
            masks[b, index] = 1
            predicts[b, index] = 2
            actions[b, index] = inputs[index]
            if key_end >= index >= key_start:
                predicts[b, index] = 1
        for index in range(inst.wlength):
            labels[b, index] = wlabels[index]
            indices[b, index] = inst.wposis[index]
            wmasks[b, index] = 1
        word_indexes.append(inst.wposis)
        b += 1

    bert_indices = torch.zeros([batch_size, max_bert_length], dtype=torch.int64, requires_grad=False)
    bert_segments = torch.zeros([batch_size, max_bert_length], dtype=torch.int64, requires_grad=False)
    bert_pieces = torch.zeros([batch_size, max_length, max_bert_length], dtype=torch.float, requires_grad=False)

    shift_pos = 1  # remove the first token
    for b in range(batch_size):
        for index in range(bert_lengths[b]):
            bert_indices[b, index] = bert_token_indices[b][index]
            bert_segments[b, index] = bert_segments_ids[b][index]

        for sindex in range(lengths[b]):
            avg_score = 1.0 / len(bert_piece_ids[b][sindex+shift_pos])
            for tindex in bert_piece_ids[b][sindex+shift_pos]:
                bert_pieces[b, sindex, tindex] = avg_score

    bert_inputs = (bert_indices, bert_segments, bert_pieces)

    return bert_inputs, actions, predicts, masks, word_indexes, indices, wmasks, labels


def batch_variable_inst(inputs, labels, vocab):
    for input, label in zip(inputs, labels):
        predicted_wlabels = []
        for idx in range(input.wlength):
            if idx < input.wkey_start or idx > input.wkey_end:
                predicted_wlabels.append(vocab.id2label(label[idx]))
            else:
                predicted_wlabels.append(input.labels[idx])
        normed_wlabels, modifies = normalize_labels(predicted_wlabels)

        normed_labels = [input.words[idx].label for idx in range(input.length)]
        for idx in range(input.wlength):
            windex = input.wposis[idx]
            normed_labels[windex] = normed_wlabels[idx]

        tokens = []
        for idx in range(input.length):
            tokens.append(Word(idx, input.words[idx].org_form, normed_labels[idx]))
        yield Sentence(tokens)
