# -*- coding: utf-8 -*-
from collections import Counter
from data.Vocab import *
import numpy as np
import codecs

def read_corpus(file_path):
    data = []
    with codecs.open(file_path, encoding='utf8') as input_file:
        for line in input_file.readlines():
            sent = line.strip().split(' ')
            if len(sent) == 0:
                print("an empty sentence, please check")
                continue
            data.append(sent)
    return data

def read_tgt_words(file_path):
    tgt_words = []
    with codecs.open(file_path, encoding='utf8') as input_file:
        for line in input_file.readlines():
            sent = line.strip().split(' ')
            if len(sent) != 1:
                print("more than one word?")
                continue
            tgt_words.append(sent[0])
    return tgt_words

def read_training_corpus(file_src_path, file_tgt_path, max_src_length, max_tgt_length):
    src_data_org = read_corpus(file_src_path)
    tgt_data_org = read_corpus(file_tgt_path)
    length_src = len(src_data_org)
    length_tgt = len(src_data_org)
    if length_src != length_tgt:
        print("The numbers of training sentences do not match")
    src_data, tgt_data = [], []
    for index in range(length_src):
        if len(src_data_org[index]) > max_src_length or len(tgt_data_org[index]) > max_tgt_length:
            continue
        src_data.append(src_data_org[index])
        tgt_data.append(tgt_data_org[index])

    return src_data, tgt_data


def creat_vocabularies(src_data, tgt_data, src_vocab_size, tgt_vocab_size):
    src_word_counter = Counter()
    for sentence in src_data:
        for word in sentence:
            src_word_counter[word] += 1

    tgt_word_counter = Counter()
    for sentence in tgt_data:
        for word in sentence:
            tgt_word_counter[word] += 1

    src_most_common = [ite for ite, it in src_word_counter.most_common(src_vocab_size)]
    tgt_most_common = [ite for ite, it in tgt_word_counter.most_common(tgt_vocab_size)]

    src_vocab = NMTVocab(src_most_common)
    tgt_vocab = NMTVocab(tgt_most_common)

    return src_vocab, tgt_vocab


def read_references_corpus(file_names):
    data_all = []
    for onefile in file_names:
        data_all.append(read_corpus(onefile))

    sent_num = len(data_all[0])
    refer_num = len(data_all)

    data_all_reversed = []
    for index in range(sent_num):
        sentences = []
        for idx in range(refer_num):
            sentences.append(data_all[idx][index])
        data_all_reversed.append(sentences)

    return data_all_reversed


def create_train_batch_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """
    bucket_len = len(data)
    if shuffle:
        for idx in range(bucket_len):
            if shuffle: np.random.shuffle(data[idx])

    batched_data = []
    for idx in range(bucket_len):
        data_size = len(data[idx])
        batch_num = int(np.ceil(data_size / float(batch_size)))
        for i in range(batch_num):
            cur_batch_size = batch_size if i < batch_num - 1 else data_size - batch_size * i
            instances = [data[idx][i * batch_size + b] for b in range(cur_batch_size)]
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(instances[src_id][0]), reverse=True)
            sorted_instances = [instances[src_id] for src_id in src_ids]
            batched_data.append(sorted_instances)

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def create_batch_iter(data, batch_size):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """
    data_size = len(data)
    src_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id][0]), reverse=True)

    data = [data[src_id] for src_id in src_ids]

    batched_data = []
    instances = []
    last_length = 0
    for instance in data:
        cur_length = len(instance[0])
        if last_length > 0 and cur_length != last_length and len(instances) > 0:
            batched_data.append(instances)
            instances = []
        instances.append(instance)
        last_length = cur_length
        if len(instances) == batch_size:
            batched_data.append(instances)
            instances = []

    if len(instances) > 0:
        batched_data.append(instances)

    for batch in batched_data:
        yield batch