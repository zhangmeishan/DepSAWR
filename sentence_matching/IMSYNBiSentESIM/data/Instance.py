class Instance:
    def __init__(self, src_words, tgt_words, tag):
        self.src_words = src_words
        self.src_forms = [curword for curword in src_words]
        self.tgt_words = tgt_words
        self.tgt_forms = [curword for curword in tgt_words]
        self.tag = tag

    def __str__(self):
        ## print source words
        output = ' '.join(self.src_words) + '\n' + ' '.join(self.tgt_words) + '\n' + self.tag + '\n'
        return output

    @property
    def src_len(self):
        return len(self.src_words)

    @property
    def tgt_len(self):
        return len(self.tgt_words)


def parseInstance(texts):
    if len(texts) != 3: return None
    src_words, tgt_words = texts[0].strip().split(' '), texts[1].strip().split(' ')
    tag = texts[2].strip()

    return Instance(src_words, tgt_words, tag)

def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
