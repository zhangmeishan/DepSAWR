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
                tag = divides[0].strip()
                cur_data = Instance(words, tag)
                data.append(cur_data)
    return data


def creatVocab(corpusFile, min_occur_count):
    word_counter = Counter()
    tag_counter = Counter()
    alldatas = read_corpus(corpusFile)
    for inst in alldatas:
        for curword in inst.forms:
            word_counter[curword] += 1
        tag_counter[inst.tag] += 1

    return SAVocab(word_counter, tag_counter, min_occur_count)


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
    length = len(batch[0].forms)
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b].forms) > length: length = len(batch[b].forms)

    masks = torch.zeros([batch_size, length], dtype=torch.float, requires_grad=False)
    tags = torch.zeros([batch_size], dtype=torch.int64, requires_grad=False)
    words = []
    lengths = []

    b = 0
    for inputs, tagid, inst in insts_numberize(batch, vocab):
        index = 0
        length = len(inputs)
        lengths.append(length)
        tags[b] = tagid
        for curword in inputs:
            masks[b, index] = 1
            index += 1

        words.append(inst.forms)

        b += 1

    return words, tags, masks


def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.words, pred_tag), pred_tag == inst.tag
