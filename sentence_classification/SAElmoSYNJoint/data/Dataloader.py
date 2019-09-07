from collections import Counter
from data.Vocab import *
from data.SA import *
import numpy as np
import torch
from torch.autograd import Variable

def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as infile:
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

def insts_numberize(insts, vocab, dep_vocab):
    for inst in insts:
        yield inst2id(inst, vocab, dep_vocab)

def inst2id(inst, vocab, dep_vocab):
    inputs = []
    for form in inst.forms:
        wordid = vocab.word2id(form)
        extwordid = vocab.extword2id(form)
        dep_wordid = dep_vocab.word2id(form)
        dep_extwordid = dep_vocab.extword2id(form)
        inputs.append([wordid, extwordid, dep_wordid, dep_extwordid])

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


def batch_data_variable(batch, vocab, dep_vocab):
    length = len(batch[0].forms)
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b].forms) > length: length = len(batch[b].forms)

    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    dep_words = Variable(torch.LongTensor(batch_size, length + 1).zero_(), requires_grad=False)
    dep_extwords = Variable(torch.LongTensor(batch_size, length + 1).zero_(), requires_grad=False)
    dep_masks = Variable(torch.Tensor(batch_size, length + 1).zero_(), requires_grad=False)
    tags = torch.LongTensor(batch_size).zero_()
    words = []
    lengths = []

    b = 0
    for inputs, tagid, inst in insts_numberize(batch, vocab, dep_vocab):
        index = 0
        length = len(inputs)
        lengths.append(length)
        tags[b] = tagid
        dep_words[b, 0], dep_extwords[b, 0], dep_masks[b, 0] = dep_vocab.ROOT, dep_vocab.ROOT, 1
        for curword in inputs:
            masks[b, index] = 1
            dep_words[b, index+1] = curword[2]
            dep_extwords[b, index+1] = curword[3]
            dep_masks[b, index + 1] = 1
            index += 1

        words.append(inst.forms)
        
        b += 1

    return words, masks, dep_words, dep_extwords, dep_masks, tags

def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.words, pred_tag), pred_tag == inst.tag
