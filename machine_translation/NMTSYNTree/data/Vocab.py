import json
import re


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


class RELVocab(object):
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 3

    def __init__(self, dict_path):

        self.dict_path = dict_path
        self._max_n_rels = -1

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])

    @property
    def max_n_rels(self):

        if self._max_n_rels == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_rels

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
        (rel_id, rel_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):
            with open(path, encoding='utf-8') as f:
                _dict = json.load(f)
                # Relation to rel index and rel frequence.
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

    def rel2id(self, rel):
        if rel in self._token2id_feq and self._token2id_feq[rel][0] < self.max_n_rels:
            return self._token2id_feq[rel][0]
        else:
            return self.UNK

    def id2rel(self, rel_id):

        return self._id2token[rel_id]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf-8') as file:
            for id, rel in enumerate(self._id2token):
                if id > self.UNK: file.write(rel + '\n')

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