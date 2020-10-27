from collections import Counter
from data.Vocab import *
from data.SA import *
import numpy as np
import torch
import codecs


def read_corpus(file_path):
    data = []
    with codecs.open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            divides = line.strip().split('|||')
            section_num = len(divides)
            if section_num == 2:
                words = divides[1].strip().split(' ')
                if len(words) >= 200: words = words[:200]
                tag = divides[0].strip()
                cur_data = Instance(words, tag)
                data.append(cur_data)
    return data


def creatVocab(corpusFile, bert_vocab_file, min_occur_count):
    word_counter = Counter()
    tag_counter = Counter()
    alldatas = read_corpus(corpusFile)
    for inst in alldatas:
        for curword in inst.forms:
            items = curword.split('##')
            if len(items) == 3 and (items[0] == 'arc' or items[0] == 'pop'):
                word_counter[curword] += 1
        tag_counter[inst.tag] += 1

    return SAVocab(word_counter, tag_counter, bert_vocab_file, min_occur_count)


def insts_numberize(insts, vocab):
    for inst in insts:
        yield inst2id(inst, vocab)


def inst2id(inst, vocab):
    inputs = []
    for curword in inst.forms:
        wordid = vocab.word2id(curword)
        inputs.append([wordid])

    return inputs, vocab.tag2id(inst.tag), inst


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
    lengths = [len(batch[b].forms) for b in range(batch_size)]
    max_length = lengths[0]
    for b in range(1, batch_size):
        if lengths[b] > max_length: max_length = lengths[b]

    actions = torch.zeros([batch_size, max_length], dtype=torch.int64, requires_grad=False)
    masks = torch.zeros([batch_size, max_length], dtype=torch.float, requires_grad=False)
    bert_lengths = []

    bert_token_indices, bert_segments_ids, bert_piece_ids = [], [], []
    tags = torch.zeros([batch_size], dtype=torch.int64, requires_grad=False)

    word_indexes = []
    b, max_bert_length = 0, 0
    for inputs, tagid, inst in insts_numberize(batch, vocab):
        for curword in inputs:
            actions[b, index] = curword[0]
            masks[b, index] = 1
            index += 1

        forms, indexes = [], []
        for index, curword in enumerate(inst.forms):
            items = curword.split('##')
            if len(items) != 3 or (items[0] != 'arc' and items[0] != 'pop'):
                forms.append(curword)
                indexes.append(index)

        word_indexes.append(indexes)

        bert_indice, segments_id, piece_id = vocab.bert_ids(' '.join(forms))

        cur_length = len(bert_indice)
        bert_lengths.append(cur_length)
        if cur_length > max_bert_length: max_bert_length = cur_length
        bert_token_indices.append(bert_indice)
        bert_segments_ids.append(segments_id)
        bert_piece_ids.append(piece_id)

        tags[b] = tagid
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
            avg_score = 1.0 / len(bert_piece_ids[b][sindex + shift_pos])
            for tindex in bert_piece_ids[b][sindex + shift_pos]:
                bert_pieces[b, sindex, tindex] = avg_score

    bert_inputs = (bert_indices, bert_segments, bert_pieces)

    return bert_inputs, actions, word_indexes, tags, masks


def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.words, pred_tag), pred_tag == inst.tag
