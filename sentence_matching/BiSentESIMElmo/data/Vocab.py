import numpy as np
class Vocab(object):
    ##please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3
    def __init__(self, word_counter, tag_counter, min_occur_count = 1):
        self._id2word = ['<pad>', '<bos>', '<eos>', '<unk>']
        self._wordid2freq = [10000, 10000, 10000, 10000]
        self._id2tag = []
        for word, count in word_counter.most_common():
            if count <= min_occur_count: continue
            self._id2word.append(word)
            self._wordid2freq.append(count)

        for tag, count in tag_counter.most_common():
            self._id2tag.append(tag)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            print("serious bug: POS tags dumplicated, please check!")

        print("Vocab info: #words %d, #tags %d" % (self.vocab_size, self.tag_size))

    def load_initialize_embs(self, embfile):
        embeddings = {}
        with open(embfile, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self._word2id:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = self.vocab_size
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self._word2id.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "<pad>":
                    assert(i == self.PAD)
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
            embedding_matrix[i] = embedding_matrix[i] / np.std(embedding_matrix[i])
        hit_count = num_words - missed
        print("Captured words: %d, total words: %d, ratio: %.f" %(hit_count, num_words, \
                                                                  hit_count*100.0/num_words))

        return embedding_matrix

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        self._id2extword = []
        allwords = set()
        for special_word in ['<pad>', '<bos>', '<eos>', '<unk>']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2extword.append(special_word)

        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) > 10:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2extword.append(curword)
                    embedding_dim = len(values) - 1
        word_num = len(self._id2extword)
        print('Total words: ' + str(word_num) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: words dumplicated, please check!")

        oov_id = self._extword2id.get('<unk>')
        if self.UNK != oov_id:
            print("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) == embedding_dim + 1:
                    index = self._extword2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    vector = vector / np.std(vector)
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
        embeddings[self.UNK] = embeddings[self.UNK] / word_num
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

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def tag_size(self):
        return len(self._id2tag)
