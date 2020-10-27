from data.Vocab import *
from data.Instance import *
from data.TensorInstances import *
from collections import Counter
import codecs


def read_corpus(file):
    data = []
    with codecs.open(file, encoding='utf8') as input_file:
        curtext = []
        for line in input_file.readlines():
            line = line.strip()
            if line is not None and line != '':
                curtext.append(line)
            else:
                slen = len(curtext)
                if slen == 3:
                    cur_data = parseInstance(curtext)
                    if cur_data.src_len <= 200 and cur_data.tgt_len <= 200:
                        data.append(cur_data)
                curtext = []

    slen = len(curtext)
    if slen == 3:
        cur_data = parseInstance(curtext)
        if cur_data.src_len <= 200 and cur_data.tgt_len <= 200:
            data.append(cur_data)

    print("Total num: " + str(len(data)))
    return data


def creatVocab(corpusFile, bert_vocab_file, min_occur_count):
    word_counter = Counter()
    tag_counter = Counter()
    alldatas = read_corpus(corpusFile)
    for inst in alldatas:
        for curword in inst.src_forms:
            word_counter[curword] += 1
        for curword in inst.tgt_forms:
            word_counter[curword] += 1
        tag_counter[inst.tag] += 1

    return BSVocab(word_counter, tag_counter, bert_vocab_file, min_occur_count)


def insts_numberize(insts, vocab, dep_vocab):
    for inst in insts:
        yield inst2id(inst, vocab, dep_vocab)


def inst2id(inst, vocab, dep_vocab):
    tagid = vocab.tag2id(inst.tag)

    src_dep_ids = dep_vocab.word2id(inst.src_forms)
    tgt_dep_ids = dep_vocab.word2id(inst.tgt_forms)
    src_dep_extids = dep_vocab.extword2id(inst.src_forms)
    tgt_dep_extids = dep_vocab.extword2id(inst.tgt_forms)

    return tagid, inst, src_dep_ids, tgt_dep_ids, src_dep_extids, tgt_dep_extids


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
    slen, tlen = len(batch[0].src_words), len(batch[0].tgt_words)
    batch_size = len(batch)
    src_lengths, tgt_lengths = [slen], [tlen]
    for b in range(1, batch_size):
        cur_slen, cur_tlen = len(batch[b].src_words), len(batch[b].tgt_words)
        if cur_slen > slen: slen = cur_slen
        if cur_tlen > tlen: tlen = cur_tlen
        src_lengths.append(cur_slen)
        tgt_lengths.append(cur_tlen)

    src_bert_token_indices, src_bert_segments_ids, src_bert_piece_ids = [], [], []
    tgt_bert_token_indices, tgt_bert_segments_ids, tgt_bert_piece_ids = [], [], []
    src_bert_lengths, tgt_bert_lengths = [], []

    b, sblen, tblen = 0, 0, 0
    for tagid, inst, src_dep_ids, tgt_dep_ids, src_dep_extids, tgt_dep_extids \
            in insts_numberize(batch, vocab):
        src_bert_indice, src_segments_id, src_piece_id = vocab.bert_ids(inst.src_sentence)
        cur_src_length = len(src_bert_indice)
        src_bert_lengths.append(cur_src_length)
        if cur_src_length > sblen: sblen = cur_src_length
        src_bert_token_indices.append(src_bert_indice)
        src_bert_segments_ids.append(src_segments_id)
        src_bert_piece_ids.append(src_piece_id)

        tgt_bert_indice, tgt_segments_id, tgt_piece_id = vocab.bert_ids(inst.tgt_sentence)
        cur_tgt_length = len(tgt_bert_indice)
        tgt_bert_lengths.append(cur_tgt_length)
        if cur_tgt_length > tblen: tblen = cur_tgt_length
        tgt_bert_token_indices.append(tgt_bert_indice)
        tgt_bert_segments_ids.append(tgt_segments_id)
        tgt_bert_piece_ids.append(tgt_piece_id)

        b += 1

    tinst = TensorInstances(batch_size, slen, tlen, sblen, tblen)

    b, shift_pos = 0, 1  # remove the first token
    for tagid, inst, src_dep_ids, tgt_dep_ids, src_dep_extids, tgt_dep_extids \
            in insts_numberize(batch, vocab):
        for index in range(src_bert_lengths[b]):
            tinst.src_bert_indices[b, index] = src_bert_token_indices[b][index]
            tinst.src_bert_segments[b, index] = src_bert_segments_ids[b][index]

        for src_sindex in range(src_lengths[b]):
            avg_score = 1.0 / len(src_bert_piece_ids[b][src_sindex+shift_pos])
            tinst.src_masks[b, src_sindex] = 1
            for src_tindex in src_bert_piece_ids[b][src_sindex+shift_pos]:
                tinst.src_bert_pieces[b, src_sindex, src_tindex] = avg_score

        for index in range(tgt_bert_lengths[b]):
            tinst.tgt_bert_indices[b, index] = tgt_bert_token_indices[b][index]
            tinst.tgt_bert_segments[b, index] = tgt_bert_segments_ids[b][index]

        for tgt_sindex in range(tgt_lengths[b]):
            avg_score = 1.0 / len(tgt_bert_piece_ids[b][tgt_sindex+shift_pos])
            tinst.tgt_masks[b, tgt_sindex] = 1
            for tgt_tindex in tgt_bert_piece_ids[b][tgt_sindex+shift_pos]:
                tinst.tgt_bert_pieces[b, tgt_sindex, tgt_tindex] = avg_score

        cur_slen, cur_tlen = len(src_dep_ids), len(tgt_dep_ids)
        for index in range(cur_slen):
            tinst.src_dep_words[b, index+1] = src_dep_ids[index]
            tinst.src_dep_extwords[b, index + 1] = src_dep_extids[index]
            tinst.src_dep_masks[b, index + 1] = 1

        for index in range(cur_tlen):
            tinst.tgt_dep_words[b, index+1] = tgt_dep_ids[index]
            tinst.tgt_dep_extwords[b, index + 1] = tgt_dep_extids[index]
            tinst.tgt_dep_masks[b, index + 1] = 1

        tinst.tags[b] = tagid
        tinst.src_lens[b] = src_lengths[b]
        tinst.tgt_lens[b] = tgt_lengths[b]
        b += 1

    return tinst


def batch_variable_inst(insts, tagids, vocab):
    for inst, tagid in zip(insts, tagids):
        pred_tag = vocab.id2tag(tagid)
        yield Instance(inst.src_words, inst.tgt_words, pred_tag), pred_tag == inst.tag
