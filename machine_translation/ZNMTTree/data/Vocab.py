import sys
sys.path.extend(["../","./"])

class SRCVocab:
    PAD, ROOT, UNK = 0, 1, 2
    def __init__(self, word_list, rel_list, relroot='root'):
        self.root = relroot
        self.root_form = '<' + relroot.lower() + '>'
        self.i2w = ['<pad>', self.root_form, '<unk>']  + word_list
        self.i2r = ['<pad>', relroot, '<unk>']

        for rel in rel_list:
            if rel != relroot: self.i2r.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        self.r2i = reverse(self.i2r)
        if len(self.r2i) != len(self.i2r):
            print("serious bug: relation labels dumplicated, please check!")


        print("Source Vocab info: #words %d, #rels %d" % (self.word_size, self.rel_size))

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self.r2i.get(x, self.UNK) for x in xs]
        return self.r2i.get(xs, self.UNK)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self.i2r[x] for x in xs]
        return self.i2r[xs]

    @property
    def word_size(self):
        return len(self.i2w)

    @property
    def rel_size(self):
        return len(self.i2r)

class TGTVocab:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    S_PAD, S_BOS, S_EOS, S_UNK = '<pad>', '<s>', '</s>', '<unk>'
    def __init__(self, word_list):
        """
        :param word_list: list of words
        """
        self.i2w = [self.S_PAD, self.S_BOS, self.S_EOS, self.S_UNK] + word_list

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        print("Target Vocab info: #words %d" % (self.size))


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf8') as file:
            for id, word in enumerate(self.i2w):
                if id > self.UNK: file.write(word + '\n')
            file.close()

    @property
    def size(self):
        return len(self.i2w)

import argparse
import pickle

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--infile', default='ch-en-model/tgt_vocab')
    argparser.add_argument('--outfile', default='ch-en-model/tgt_vocab.txt')

    args, extra_args = argparser.parse_known_args()

    vocab = pickle.load(open(args.infile, 'rb'))
    vocab.save2file(args.outfile)
