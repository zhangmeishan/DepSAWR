class Instance:
    def __init__(self, words, tag):
        self.words = words
        self.forms = [word for word in words]
        self.tag = tag
        self.sentence = ' '.join(self.words)

    def __str__(self):
        output = self.tag + "|||" + self.sentence
        return output


def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')


def printInstance(output, inst):
    output.write(str(inst) + '\n')
