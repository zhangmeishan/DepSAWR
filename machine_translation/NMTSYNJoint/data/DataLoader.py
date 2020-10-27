# -*- coding: utf-8 -*-
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


def create_train_batch_iter(data, batch_size, bacthing_key='sample', shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    def _fill_buffer(b_idx, b_i):
        if b_idx == -1:
            raise IndexError

        cur_size = 0

        if bacthing_key == 'sample':
            for idx in range(b_idx, bucket_len):
                data_size = len(data[idx])
                batch_num = int(np.ceil(data_size / float(batch_size)))
                for i in range(b_i, batch_num):
                    cur_batch_size = batch_size if i < batch_num - 1 else data_size - batch_size * i
                    instances = [data[idx][i * batch_size + b] for b in range(cur_batch_size)]
                    src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(instances[src_id][0]), reverse=True)
                    sorted_instances = [instances[src_id] for src_id in src_ids]
                    buffer.append(sorted_instances)
                    cur_size += 1

                    if cur_size >= buffer_size:
                        return idx, i + 1

        elif bacthing_key == 'tokens':
            max_len = 0
            cur_batch_size = 0
            instances = []

            for idx in range(b_idx, bucket_len):
                data_size = len(data[idx])

                for i in range(b_i, data_size):
                    instances.append(data[idx][i])
                    cur_batch_size += 1
                    max_len = max(len(data[idx][i][0]), len(data[idx][i][1]), max_len)
                    # if max_len * cur_batch_size >= batch_size:
                    if max_len * cur_batch_size >= buffer_size:
                        src_ids = sorted(range(cur_batch_size),
                                         key=lambda src_id: len(instances[src_id][0]) + np.random.uniform(-1, 1),
                                         reverse=True)
                        sorted_instances = [instances[src_id] for src_id in src_ids]

                        max_len = 0
                        num_samples = 0
                        tmp = []
                        for line in sorted_instances:
                            num_samples += 1
                            max_len = max(max(len(line[0]), len(line[1])), max_len)
                            tmp.append(line)
                            if max_len * num_samples >= batch_size:
                                buffer.append(tmp)
                                max_len = 0
                                num_samples = 0
                                tmp = []

                        if len(tmp) != 0:
                            buffer.append(tmp)

                        return idx, i + 1

            if len(instances) > 0:
                src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(instances[src_id][0]), reverse=True)
                sorted_instances = [instances[src_id] for src_id in src_ids]

                max_len = 0
                num_samples = 0
                tmp = []
                for line in sorted_instances:
                    num_samples += 1
                    max_len = max(len(line[0]), len(line[1]), max_len)
                    tmp.append(line)
                    if max_len * num_samples >= batch_size:
                        buffer.append(tmp)
                        max_len = 0
                        num_samples = 0
                        tmp = []

                if len(tmp) != 0:
                    buffer.append(tmp)

                return idx, i + 1

        else:
            raise NameError('No batching key {}, please choose the tokens or sample'.format(bacthing_key))

        if len(buffer) <= 0:
            raise IndexError
        else:
            return -1, -1

    # print('creating train iterator......')
    bucket_len = len(data)
    if bacthing_key == 'sample':
        buffer_size = 20 * batch_size
    else:
        buffer_size = 200 * batch_size
    buffer = []

    if shuffle:
        print('begin shuffle .......')
        for idx in range(bucket_len):
            if shuffle: np.random.shuffle(data[idx])
        print('shuffle done')

    b_idx, b_i = 0, 0
    while True:
        if len(buffer) == 0:
            try:
                b_idx, b_i = _fill_buffer(b_idx, b_i)
            except IndexError:
                break

        yield buffer.pop(0)


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
