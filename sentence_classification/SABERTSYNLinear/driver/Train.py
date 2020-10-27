import sys

sys.path.extend(["../", "./"])
import time
import torch.optim.lr_scheduler
import random
import argparse
from driver.Config import *
from model.SAModel import *
from model.BertModel import *
from driver.SAHelper import *
from data.Dataloader import *
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle


def train(data, dev_data, test_data, classifier, vocab, config):
    global_step = 0
    best_acc = 0
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))

    optimizers, schedulers = [], []

    optimizer_model = Optimizer(filter(lambda p: p.requires_grad, classifier.model.parameters()), config)
    optimizers.append(optimizer_model)

    if config.bert_tune == 1:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in classifier.bert.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier.bert.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer_bert = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0,
                                                         num_training_steps=4000)
        optimizers.append(optimizer_bert)
        schedulers.append(scheduler_bert)

    update_freq = config.validate_every * config.update_every
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter) + ', total batch num: ' + str(batch_num))
        batch_iter = 0

        correct_num, total_num, loss_value = 0, 0, 0
        for onebatch in data_iter(data, config.train_batch_size, True):
            bert_inputs, actions, word_indexes, tags, masks = \
                batch_data_variable(onebatch, vocab)

            classifier.model.train()
            classifier.forward(bert_inputs, actions, word_indexes, masks)
            loss = classifier.compute_loss(tags)
            loss = loss / config.update_every
            loss_value += loss.item()
            loss.backward()

            cur_correct, cur_count = classifier.compute_accuracy(tags)
            correct_num += cur_correct
            total_num += cur_count
            acc = correct_num * 100.0 / total_num

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, classifier.model.parameters()), \
                                         max_norm=config.clip)
                if config.bert_tune == 1:
                    nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, classifier.bert.parameters()), \
                                             max_norm=config.clip)

                for optimizer in optimizers:
                    optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()

                classifier.model.zero_grad()
                if config.bert_tune == 1:
                    classifier.bert.zero_grad()

                during_time = float(time.time() - start_time)
                print("Step:%d, ACC:%.2f, Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                      % (global_step, acc, iter, batch_iter - 1, during_time, loss_value))

                loss_value = 0

                global_step += 1

            if batch_iter % update_freq == 0 or batch_iter == batch_num:
                tag_correct, tag_total, dev_tag_acc = \
                    evaluate(dev_data, classifier, vocab, config.dev_file + '.' + str(global_step))
                print("Dev: acc = %d/%d = %.2f" % (tag_correct, tag_total, dev_tag_acc))

                tag_correct, tag_total, test_tag_acc = \
                    evaluate(test_data, classifier, vocab, config.test_file + '.' + str(global_step))
                print("Test: acc = %d/%d = %.2f" % (tag_correct, tag_total, test_tag_acc))
                if dev_tag_acc > best_acc:
                    print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_tag_acc))
                    best_acc = dev_tag_acc
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(classifier.model.state_dict(), config.save_model_path)


def evaluate(data, classifier, vocab, outputFile):
    start = time.time()
    classifier.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    tag_correct, tag_total = 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        bert_inputs, actions, word_indexes, tags, masks = batch_data_variable(onebatch, vocab)
        count = 0
        pred_tags = classifier.classifier(bert_inputs, actions, word_indexes, masks)
        for inst, bmatch in batch_variable_inst(onebatch, pred_tags, vocab):
            printInstance(output, inst)
            tag_total += 1
            if bmatch: tag_correct += 1
            count += 1

    output.close()

    acc = tag_correct * 100.0 / tag_total

    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  classifier time = %.2f " % (len(data), during_time))

    return tag_correct, tag_total, acc


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
    torch.manual_seed(888)
    torch.cuda.manual_seed(888)
    random.seed(888)
    np.random.seed(888)

    # gpu
    gpu = torch.cuda.is_available()
    print(torch.__version__)
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    vocab = creatVocab(config.train_file, config.bert_vocab_file, config.min_occur_count)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    config.use_cuda = False
    gpu_id = -1
    if gpu and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        config.use_cuda = True
        print("GPU ID: ", args.gpu)
        gpu_id = args.gpu

    bert = BertExtractor(config)
    model = SAModel(vocab, config, bert.bert_hidden_size, bert.bert_layers)

    if config.use_cuda:
        # torch.backends.cudnn.enabled = True
        model = model.cuda()
        bert = bert.cuda()

    classifier = SentenceClassifier(model, bert, vocab)

    data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    train(data, dev_data, test_data, classifier, vocab, config)
