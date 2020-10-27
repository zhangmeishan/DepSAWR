class Instance:
    def __init__(self, src_words, src_heads, src_rels, tgt_words, tgt_heads, tgt_rels, tag):
        self.src_words = src_words
        self.src_forms = [curword for curword in src_words]
        self.src_heads = src_heads
        self.src_rels = src_rels
        self.tgt_words = tgt_words
        self.tgt_forms = [curword for curword in tgt_words]
        self.tgt_heads = tgt_heads
        self.tgt_rels = tgt_rels
        self.tag = tag
        self.src_sentence = ' '.join(self.src_words)
        self.tgt_sentence = ' '.join(self.tgt_words)

    def __str__(self):
        ## print source words
        src_items = []
        for idx in range(self.src_len):
            src_word = self.src_words[idx]
            src_head = self.src_heads[idx]
            src_rel = self.src_rels[idx]
            src_items.append(src_word + '_' + str(src_head) + '_' + src_rel)

        tgt_items = []
        for idx in range(self.tgt_len):
            tgt_word = self.tgt_words[idx]
            tgt_head = self.tgt_heads[idx]
            tgt_rel = self.tgt_rels[idx]
            tgt_items.append(tgt_word + '_' + str(tgt_head) + '_' + tgt_rel)

        output = ' '.join(src_items) + '\n' + ' '.join(tgt_items) + '\n' + self.tag + '\n'
        return output

    @property
    def src_len(self):
        return len(self.src_words)

    @property
    def tgt_len(self):
        return len(self.tgt_words)


def parseInputLine(sentline):
    worditems = sentline.strip().split(' ')
    words, heads, rels = [], [], []
    for worditem in worditems:
        id1 = worditem.rfind('_')
        id2 = worditem.rfind('_', 0, id1 - 1)
        words.append(worditem[:id2])
        heads.append(int(worditem[id2 + 1:id1]))
        rels.append(worditem[id1 + 1:])
    return words, heads, rels


def parseInstance(texts):
    if len(texts) != 3: return None
    src_words, src_heads, src_rels = parseInputLine(texts[0])
    tgt_words, tgt_heads, tgt_rels = parseInputLine(texts[1])
    tag = texts[2].strip()

    return Instance(src_words, src_heads, src_rels, tgt_words, tgt_heads, tgt_rels, tag)


def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')


def printInstance(output, inst):
    output.write(str(inst) + '\n')
