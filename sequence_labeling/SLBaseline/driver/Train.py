import sys

sys.path.extend(["../../", "../", "./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from model.BiLSTMModel import *
from driver.SLHelper import *
from data.Dataloader import *
import pickle


def train(data, dev_data, test_data, labeler, vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, labeler.model.parameters()), config)

    global_step = 0
    best_acc = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num = 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, predicts, masks, labels = \
                batch_data_variable(onebatch, vocab)

            labeler.model.train()

            labeler.forward(words, extwords, predicts, masks)
            loss = labeler.compute_loss(labels, masks)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            cur_correct, cur_count = labeler.compute_accuracy(labels)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num
            during_time = float(time.time() - start_time)
            print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                  % (global_step, acc, iter, batch_iter, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                         max_norm=config.clip)
                optimizer.step()
                labeler.model.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                tag_correct, tag_total, dev_tag_acc = \
                    evaluate(dev_data, labeler, vocab, config.dev_file + '.' + str(global_step))
                print("Dev: acc = %d/%d = %.2f" % (tag_correct, tag_total, dev_tag_acc))

                tag_correct, tag_total, test_tag_acc = \
                    evaluate(test_data, labeler, vocab, config.test_file + '.' + str(global_step))
                print("Test: acc = %d/%d = %.2f" % (tag_correct, tag_total, test_tag_acc))
                if dev_tag_acc > best_acc:
                    print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_tag_acc))
                    best_acc = dev_tag_acc
                    if iter > config.save_after > 0:
                        torch.save(labeler.model.state_dict(), config.save_model_path)


def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, predicts, masks, labels = \
            batch_data_variable(onebatch, vocab)
        count = 0
        predict_labels = labeler.labeler(words, extwords, predicts, masks)
        for result in batch_variable_inst(onebatch, predict_labels, vocab):
            printInstance(output, result)
            gold_entity_num, predict_entity_num, correct_entity_num = evalInstance(onebatch[count], result)
            total_gold_entity_num += gold_entity_num
            total_predict_entity_num += predict_entity_num
            total_correct_entity_num += correct_entity_num
            count += 1

    output.close()

    acc = total_correct_entity_num * 200.0 / (total_predict_entity_num + total_gold_entity_num)

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  labeler time = %.2f " % (len(data), during_time))

    return total_correct_entity_num, total_gold_entity_num, acc


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


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

    vocab = creat_vocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    config.use_cuda = False
    if gpu and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)

    model = BiLSTMModel(vocab, config, vec)
    if config.use_cuda:
        # torch.backends.cudnn.enabled = True
        model = model.cuda()

    labeler = SequenceLabeler(model)

    data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    train(data, dev_data, test_data, labeler, vocab, config)
