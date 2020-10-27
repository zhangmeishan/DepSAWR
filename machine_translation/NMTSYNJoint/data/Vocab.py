import json
import re
import numpy as np
import copy


class _Tokenizer(object):
    """The abstract class of Tokenizer

    Implement ```tokenize``` method to split a string of sentence into tokens.
    Implement ```detokenize``` method to combine tokens into a whole sentence.
    ```special_tokens``` stores some helper tokens to describe and restore the tokenizing.
    """

    def __init__(self):
        pass

    def tokenize(self, sent):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError


class WordTokenizer(_Tokenizer):

    def __init__(self):
        super(WordTokenizer, self).__init__()

    def tokenize(self, sent):
        return sent.strip().split()

    def detokenize(self, tokens):
        return ' '.join(tokens)


class BPETokenizer(_Tokenizer):

    def __init__(self):
        """ Byte-Pair-Encoding (BPE) Tokenizer

        Args:
            codes: Path to bpe codes. Default to None, which means the text has already been segmented  into
                bpe tokens.
        """
        super(BPETokenizer, self).__init__()

    def tokenize(self, sent):
        return sent.strip().split()

    def detokenize(self, tokens):
        return re.sub(r"@@\s|@@$", "", " ".join(tokens))
        # return ' '.join(tokens).replace("@@ ", "")


class Tokenizer(object):

    def __new__(cls, type):
        if type == "word":
            return WordTokenizer()
        elif type == "bpe":
            return BPETokenizer()
        else:
            print("Unknown tokenizer type {0}".format(type))
            raise ValueError


class NMTVocab(object):
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 3

    def __init__(self, type, dict_path, max_n_words=-1):

        self.dict_path = dict_path
        self._max_n_words = max_n_words

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.tokenizer = Tokenizer(type=type)  # type: _Tokenizer

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<pad>": (self.PAD, 0),
            "</s>": (self.EOS, 0),
            "<s>": (self.BOS, 0),
            "<unk>": (self.UNK, 0)
        }

    def _load_vocab(self, path):
        """
        Load vocabulary from file

        If file is formatted as json, for each item the key is the token, while the value is a tuple such as
        (word_id, word_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path, encoding='utf-8') as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    def word2id(self, word):
        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:
            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2word(self, word_id):

        return self._id2token[word_id]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf-8') as file:
            for id, word in enumerate(self._id2token):
                if id > self.UNK: file.write(word + '\n')

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.BOS

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.PAD

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.EOS

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.UNK


class Vocab(object):
    PAD, ROOT, UNK = 0, 1, 2

    def __init__(self, word_counter, rel_counter, relroot='root', min_occur_count = 2):
        self._root = relroot
        self._root_form = '<' + relroot.lower() + '>'
        self._id2word = ['<pad>', self._root_form, '<unk>']
        self._wordid2freq = [10000, 10000, 10000]
        self._id2extword = ['<pad>', self._root_form, '<unk>']
        self._id2rel = ['<pad>', relroot]
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for rel, count in rel_counter.most_common():
            if rel != relroot: self._id2rel.append(rel)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relation labels dumplicated, please check!")

        print("Vocab info: #words %d, #rels %d" % (self.vocab_size, self.rel_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def copyfrom(self, vocab):
       self._root = vocab._root
       self._root_form = vocab._root_form

       self._id2word = copy.deepcopy(vocab._id2word)
       self._wordid2freq = copy.deepcopy(vocab._wordid2freq)
       self._id2extword = copy.deepcopy(vocab._id2extword)
       self._id2rel = copy.deepcopy(vocab._id2rel)

       self._word2id = copy.deepcopy(vocab._word2id)
       self._extword2id = copy.deepcopy(vocab._extword2id)
       self._rel2id = copy.deepcopy(vocab._rel2id)

       print("Vocab info: #words %d, #rels %d" % (self.vocab_size, self.rel_size))


    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def rel_size(self):
        return len(self._id2rel)

