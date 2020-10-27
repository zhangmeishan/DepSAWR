# -*- coding: utf-8 -*-
import sys

sys.path.extend(["../", "./"])
import argparse
import random
from driver.Config import *
from driver.Optim import *
from driver.NMTHelper import *
from driver.Scheduler import *
from data.Vocab import NMTVocab
from model.DL4MT import DL4MT
from model.Transformer import Transformer
from module.Criterions import *
from module.Utils import Saver
from metric.BLEU import SacreBLEUScorer

import ntpath
import pickle
import os
import re


def train(nmt, train_srcs, train_tgts, config):
    global_step = 0
    nmt.prepare_training_data(train_srcs, train_tgts)
    valid_files = config.dev_files.strip().split(' ')
    # valid_srcs = read_corpus(valid_files[0])
    # valid_tgts = read_corpus(valid_files[1])
    # nmt.prepare_valid_data(valid_srcs, valid_tgts)

    optim = Optimizer(name=config.learning_algorithm,
                      model=nmt.model,
                      lr=config.learning_rate,
                      grad_clip=config.clip
                      )

    if config.schedule_method == 'noam':
        scheduler = NoamScheduler(optimizer=optim, d_model=config.embed_size, warmup_steps=8000)
    elif config.schedule_method == 'loss':
        scheduler = ReduceOnPlateauScheduler(optimizer=optim,
                                             patience=20,
                                             min_lr=5e-05,
                                             scale=0.5)

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(config.save_dir, config.model_name)),
                             num_max_keeping=config.num_kept_checkpoints
                             )

    print('start training...')
    best_bleu = -1
    mini_step = 0

    for iter in range(config.train_iters):
        # dynamic adjust lr
        total_stats = Statistics()
        iterator = create_train_batch_iter(nmt.train_data, nmt.batch_size, bacthing_key=config.batching_key,
                                           shuffle=False)
        batch_iter, total_iters = 0, nmt.train_size
        n_words_t = 0

        for batch in iterator:
            mini_step += 1

            if config.schedule_method is not None \
                    and config.schedule_method != "loss":
                scheduler.step(global_step=global_step + 1)

            batch_iter += len(batch)
            n_words_t += sum(len(s[1]) for s in batch)
            try:
                stat = nmt.train_one_batch(batch)
                total_stats.update(stat)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    optim.zero_grad()
                else:
                    raise e

            if mini_step % config.update_every == 0:
                global_step += 1
                optim.step()

                lrate = list(optim.get_lrate())[0]
                optim.zero_grad()
                total_stats.print_out(global_step - 1, iter, batch_iter, total_iters, lrate, n_words_t, best_bleu)
                n_words_t = 0

                if global_step > config.eval_start and global_step % config.validate_every == 0:
                    dev_start_time = time.time()
                    dev_bleu = evaluate(nmt, valid_files, config, global_step)
                    during_time = float(time.time() - dev_start_time)
                    print("step %d, epoch %d: dev bleu: %.2f, time %.2f" \
                          % (global_step, iter, dev_bleu, during_time))

                    if dev_bleu > best_bleu:
                        lrs = [lr for lr in optim.get_lrate()]
                        print("Exceed best bleu: history = %.2f, current = %.2f, lr_ratio = %.6f" % \
                              (best_bleu, dev_bleu, lrs[0]))
                        best_bleu = dev_bleu

                        if global_step > config.save_after:
                            checkpoint_saver.save(global_step=global_step, model=nmt.model,
                                                  optim=optim)


def evaluate(nmt, eval_files, config, global_step):
    valid_srcs = read_corpus(eval_files[0])
    eval_data = nmt.prepare_eval_data(valid_srcs)
    result = nmt.translate(eval_data)

    bleu_scorer = SacreBLEUScorer(reference_path=eval_files[1],
                                  num_refs=1,
                                  lang_pair='de-en',
                                  sacrebleu_args="--tokenize none -lc",
                                  postprocess=False
                                  )

    head, tail = ntpath.split(eval_files[1])

    outputFile = os.path.join(config.save_dir, tail + '.' + str(global_step))
    output = open(outputFile, 'w', encoding='utf-8')

    ordered_result = []
    for idx, instance in enumerate(eval_data):
        src_key = '\t'.join(instance[1])
        cur_result = result.get(src_key)
        if cur_result is not None:
            ordered_result.append(cur_result)
        else:
            print("Strange, miss one sentence")
            ordered_result.append([''])

        sentence_out = ' '.join(ordered_result[idx])
        sentence_out = sentence_out.replace(' <unk>', '')
        sentence_out = sentence_out.replace('@@ ', '')

        output.write(sentence_out + '\n')

    output.close()

    with open(outputFile, 'r', encoding='utf-8') as f:
        bleu_val = bleu_scorer.corpus_bleu(f)

    return bleu_val


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.gpu >= 0:
        config.use_cuda = True
        torch.cuda.set_device(args.gpu)
        print("GPU ID: ", args.gpu)
    print("\nGPU using status: ", config.use_cuda)

    train_files = config.train_files.strip().split(' ')
    train_srcs, train_tgts = read_training_corpus(train_files[0], train_files[1], \
                                                  config.max_src_length, config.max_tgt_length)

    src_vocab = NMTVocab(config.src_vocab_type, config.src_vocab_path)
    tgt_vocab = NMTVocab(config.tgt_vocab_type, config.tgt_vocab_path)

    print("Sentence Number: #train = %d" % (len(train_srcs)))

    # model
    nmt_model = eval(config.model_name)(config, src_vocab, tgt_vocab, config.use_cuda)
    critic = NMTCriterion(padding_idx=tgt_vocab.PAD, label_smoothing=config.label_smoothing)

    if config.use_cuda:
        # torch.backends.cudnn.enabled = False
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    nmt = NMTHelper(nmt_model, critic, src_vocab, tgt_vocab, config)

    train(nmt, train_srcs, train_tgts, config)
