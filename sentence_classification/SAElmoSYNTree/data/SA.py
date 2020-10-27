class Instance:
    def __init__(self, words, heads, rels, tag):
        self.words = words
        self.forms = [word.lower() for word in words]
        self.heads = heads
        self.rels = rels
        self.tag = tag

    def __str__(self):
        worditems = []
        for word, head, rel in zip(self.words, self.heads, self.rels):
            worditems.append(word + '_' + str(head) + '_' + rel)
        output = self.tag + "|||" + ' '.join(worditems)
        return output

def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
