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
                word_counter[token.form] += 1
                if index < sentence.key_start or index > sentence.key_end:
                    label_counter[token.label] += 1
                index = index + 1

    return SLVocab(word_counter, label_counter, bert_vocab_file, min_occur_count)


def insts_numberize(insts, vocab, dep_vocab):
    for inst in insts:
        yield inst2id(inst, vocab, dep_vocab)


def inst2id(inst, vocab, dep_vocab):
    inputs = []
    index = 0
    for curword in inst.words:
        dep_wordid = dep_vocab.word2id(curword)
        dep_extwordid = dep_vocab.extword2id(curword)
        if index < inst.key_start or index > inst.key_end:
            labelid = vocab.label2id(curword.label)
        else:
            labelid = vocab.PAD
        index = index + 1
        inputs.append([dep_wordid, dep_extwordid, labelid])

    return inputs, inst.key_start, inst.key_end, inst


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


def batch_data_variable(batch, vocab, dep_vocab):
    batch_size = len(batch)
    lengths = [len(batch[b].words) for b in range(batch_size)]
    max_length = lengths[0]
    for b in range(1, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]

    dep_words = torch.zeros([batch_size, max_length + 1], dtype=torch.int64, requires_grad=False)
    dep_extwords = torch.zeros([batch_size, max_length + 1], dtype=torch.int64, requires_grad=False)
    dep_masks = torch.zeros([batch_size, max_length + 1], dtype=torch.float, requires_grad=False)
    predicts = torch.zeros([batch_size, max_length], dtype=torch.int64, requires_grad=False)
    masks = torch.zeros([batch_size, max_length], dtype=torch.float, requires_grad=False)
    labels = torch.zeros([batch_size, max_length], dtype=torch.int64, requires_grad=False)
    bert_lengths = []

    bert_token_indices, bert_segments_ids, bert_piece_ids = [], [], []
    b, max_bert_length = 0, 0
    for inputs, key_start, key_end, inst in insts_numberize(batch, vocab, dep_vocab):
        bert_indice, segments_id, piece_id = vocab.bert_ids(inst.sentence)

        cur_length = len(bert_indice)
        bert_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length
        bert_token_indices.append(bert_indice)
        bert_segments_ids.append(segments_id)
        bert_piece_ids.append(piece_id)

        index = 0
        dep_words[b, 0], dep_extwords[b, 0], dep_masks[b, 0] = dep_vocab.ROOT, dep_vocab.ROOT, 1
        for curword in inputs:
            masks[b, index] = 1
            predicts[b, index] = 2
            dep_words[b, index+1] = curword[0]
            dep_extwords[b, index+1] = curword[1]
            dep_masks[b, index + 1] = 1
            labels[b, index] = curword[2]
            if key_end >= index >= key_start:
                predicts[b, index] = 1

            index += 1
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

    return bert_inputs, predicts, masks, dep_words, dep_extwords, dep_masks, labels


def batch_variable_inst(inputs, labels, vocab):
    for input, label in zip(inputs, labels):
        predicted_labels = []
        for idx in range(input.length):
            if idx < input.key_start or idx > input.key_end:
                predicted_labels.append(vocab.id2label(label[idx]))
            else:
                predicted_labels.append(input.words[idx].label)
        normed_labels, modifies = normalize_labels(predicted_labels)
        tokens = []
        for idx in range(input.length):
            tokens.append(Word(idx, input.words[idx].org_form, normed_labels[idx]))
        yield Sentence(tokens)

