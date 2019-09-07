import re

def tokenize(input_word):
    input_word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", input_word)
    input_word = re.sub(r"\'s", " is", input_word)
    input_word = re.sub(r"\'ve", " have", input_word)
    input_word = re.sub(r"n\'t", " not", input_word)
    input_word = re.sub(r"\'re", " are", input_word)
    input_word = re.sub(r"\'d", " had", input_word)
    input_word = re.sub(r"\'ll", " will", input_word)
    input_word = re.sub(r",", " , ", input_word)
    input_word = re.sub(r"!", " ! ", input_word)
    input_word = re.sub(r"\(", " \( ", input_word)
    input_word = re.sub(r"\)", " \) ", input_word)
    input_word = re.sub(r"\?", " \? ", input_word)
    input_word = re.sub(r"\s{2,}", " ", input_word)
    return input_word.strip().lower()


class Instance:
    def __init__(self, words, tag):
        self.words = words
        self.forms = [tokenize(word) for word in words]
        self.tag = tag

    def __str__(self):
        output = self.tag + "|||" + ' '.join(self.words)
        return output


def readInstance(file):
    total = 0
    for line in file:
        divides = line.strip().split('|||')
        section_num = len(divides)
        if section_num == 2:
            words = divides[1].strip().split(' ')
            tag = divides[0].strip()
            total += 1
            yield Instance(words, tag)
        else:
            pass

    print("Total num: ", total)


def writeInstance(filename, insts):
    with open(filename, 'w') as file:
        for inst in insts:
            file.write(str(inst) + '\n')

def printInstance(output, inst):
    output.write(str(inst) + '\n')
