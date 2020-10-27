import sys

sys.path.extend(["../../", "../", "./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.ParserConfig import *
from model.BiLSTMModel import *
from model.ParserModel import *
from driver.SLHelper import *
from data.Dataloader import *
import pickle


def train(data, dev_data, test_data, labeler, vocab, dep_vocab, config):
    optimizers = []
    optimizers.append(Optimizer(labeler.model, config))
    if config.parser_tune == 1:
        optimizers.append(Optimizer(labeler.parser, config))

    global_step = 0
    best_acc = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    update_freq = config.validate_every * config.update_every
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, predicts, masks, dep_words, dep_extwords, dep_masks, labels = \
                batch_data_variable(onebatch, vocab, dep_vocab)

            labeler.model.train()

            labeler.forward(words, extwords, predicts, masks, dep_words, dep_extwords, dep_masks)
            loss = labeler.compute_loss(labels, masks)
            loss = loss / config.update_every
            loss_value += loss.item()
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
                for optimizer in optimizers:
                    optimizer.step(config.clip)
                labeler.model.zero_grad()
                if config.parser_tune == 1: labeler.parser.zero_grad()
                during_time = float(time.time() - start_time)
                print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                      % (global_step, acc, iter, batch_iter - 1, during_time, loss_value))
                loss_value = 0

                global_step += 1

            if batch_iter % update_freq == 0 or batch_iter == batch_num:
                tag_correct, tag_total, dev_tag_acc = \
                    evaluate(dev_data, labeler, vocab, dep_vocab, config.dev_file + '.' + str(global_step))
                print("Dev: acc = %d/%d = %.2f" % (tag_correct, tag_total, dev_tag_acc))

                tag_correct, tag_total, test_tag_acc = \
                    evaluate(test_data, labeler, vocab, dep_vocab, config.test_file + '.' + str(global_step))
                print("Test: acc = %d/%d = %.2f" % (tag_correct, tag_total, test_tag_acc))
                if dev_tag_acc > best_acc:
                    print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_tag_acc))
                    best_acc = dev_tag_acc
                    if iter > config.save_after > 0:
                        torch.save(labeler.model.state_dict(), config.save_model_path)


def evaluate(data, labeler, vocab, dep_vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, predicts, masks, dep_words, dep_extwords, dep_masks, labels = \
            batch_data_variable(onebatch, vocab, dep_vocab)
        count = 0
        predict_labels = labeler.labeler(words, extwords, predicts, masks,
                                         dep_words, dep_extwords, dep_masks)
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
    def __init__(self, model, config):
        # Get all parameters that require grads
        self.params = self.get_params(model)
        self.optim = torch.optim.Adam(self.params, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    @staticmethod
    def get_params(model):
        """Returns all name, parameter pairs with requires_grad=True."""
        return list(
            filter(lambda p: p.requires_grad, model.parameters()))

    def step(self, gclip=-1):
        self.clip(gclip)
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def clip(self, gclip):
        if gclip > 0:
            nn.utils.clip_grad_norm_(self.params, max_norm=gclip)

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
    argparser.add_argument('--parser_config_file', default='parser.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    parser_config = ParserConfigurable(args.parser_config_file)

    dep_vocab = pickle.load(open(parser_config.load_vocab_path, 'rb'))
    dep_vec = dep_vocab.create_placeholder_embs(parser_config.pretrained_embeddings_file)

    parser_model = ParserModel(dep_vocab, parser_config, dep_vec)
    parser_model.load_state_dict(torch.load(parser_config.load_model_path, \
                                            map_location=lambda storage, loc: storage))

    vocab = creat_vocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    config.use_cuda = False
    gpu_id = -1
    if gpu and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)
        gpu_id = args.gpu

    model = BiLSTMModel(vocab, config, parser_config, vec)
    if config.use_cuda:
        # torch.backends.cudnn.enabled = True
        model = model.cuda()
        parser_model = parser_model.cuda()

    labeler = SequenceLabeler(model, parser_model)

    data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    train(data, dev_data, test_data, labeler, vocab, dep_vocab, config)
