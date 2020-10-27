class Word:
    def __init__(self, id, form, label):
        self.id = id
        self.org_form = form
        self.form = form.lower()
        self.label = label
        # 1 indicates word, 0 indicates syn
        self.wtype = 0 if label == "###" else 1

    def __str__(self):
        values = [str(self.id), self.org_form, self.label]
        return '\t'.join(values)


class Sentence:
    def __init__(self, words):
        self.words = list(words)
        self.length = len(self.words)
        self.key_head = -1
        self.key_start = -1
        self.key_end = -1
        self.key_label = ""
        self.span = False

        self.wkey_head = -1
        self.wkey_start = -1
        self.wkey_end = -1

        self.wlength, self.forms, self.labels = 0, [], []
        self.wposis, self.r_wposis = [], []
        for idx in range(self.length):
            if words[idx].wtype == 1:
                self.wlength = self.wlength + 1
                self.forms.append(words[idx].org_form)
                self.labels.append(words[idx].label)
                num_words = len(self.wposis)
                self.r_wposis.append(num_words)
                self.wposis.append(idx)
            else:
                self.r_wposis.append(-1)
        self.sentence = ' '.join(self.forms)

        for idx in range(self.length):
            if words[idx].label.endswith("-*"):
                self.key_head = idx
                self.wkey_head = self.r_wposis[idx]
                self.key_label = words[idx].label[2:-2]
                break

        if self.key_head != -1:
            self.span = True
            for idx in range(self.length):
                cur_label = words[idx].label
                if cur_label.startswith("B-"+self.key_label) \
                        or cur_label.startswith("S-"+self.key_label):
                    self.key_start = idx
                    self.wkey_start = self.r_wposis[idx]
                if cur_label.startswith("E-"+self.key_label) \
                        or cur_label.startswith("S-"+self.key_label):
                    self.key_end = idx
                    self.wkey_end = self.r_wposis[idx]
        else:
            self.key_start, self.wkey_start = self.length, self.wlength
            self.key_end, self.wkey_end = -1, -1


def label_to_entity(labels):
    length = len(labels)
    entities = set()
    idx = 0
    while idx < length:
        if labels[idx] == "O":
            idx = idx + 1
        elif labels[idx].startswith("B-"):
            label = labels[idx][2:]
            predict = False
            if label.endswith("-*"):
                label = label[0:-2]
                predict = True
            next_idx = idx + 1
            end_idx = idx
            while next_idx < length:
                if labels[next_idx] == "O" or labels[next_idx].startswith("B-") \
                        or labels[next_idx].startswith("S-"):
                    break
                next_label = labels[next_idx][2:]
                if next_label.endswith("-*"):
                    next_label = next_label[0:-2]
                    predict = True
                if next_label != label:
                    break
                end_idx = next_idx
                next_idx = next_idx + 1
            if end_idx == idx:
                new_label = "S-" + labels[idx][2:]
                print("Change %s to %s" % (labels[idx], new_label))
                labels[idx] = new_label
            if not predict:
                entities.add("[%d,%d]%s"%(idx, end_idx, label))
            idx = end_idx + 1
        elif labels[idx].startswith("S-"):
            label = labels[idx][2:]
            predict = False
            if label.endswith("-*"):
                label = label[0:-2]
                predict = True
            if not predict:
                entities.add("[%d,%d]%s"%(idx, idx, label))
            idx = idx + 1
        elif labels[idx].startswith("M-"):
            new_label = "B-" + labels[idx][2:]
            print("Change %s to %s" % (labels[idx], new_label))
            labels[idx] = new_label
        else:
            new_label = "S-" + labels[idx][2:]
            print("Change %s to %s" % (labels[idx], new_label))
            labels[idx] = new_label

    return entities


def normalize_labels(labels):
    length = len(labels)
    change = 0
    normed_labels = []
    for idx in range(length):
        normed_labels.append(labels[idx])
    idx = 0
    while idx < length:
        if labels[idx] == "O":
            idx = idx + 1
        elif labels[idx].startswith("B-"):
            label = labels[idx][2:]
            if label.endswith("-*"):
                label = label[0:-2]
            next_idx = idx + 1
            end_idx = idx
            while next_idx < length:
                if labels[next_idx] == "O" or labels[next_idx].startswith("B-") \
                        or labels[next_idx].startswith("S-"):
                    break
                next_label = labels[next_idx][2:]
                if next_label.endswith("-*"):
                    next_label = next_label[0:-2]
                if next_label != label:
                    break
                end_idx = next_idx
                next_idx = next_idx + 1
            if end_idx == idx:
                new_label = "S-" + labels[idx][2:]
                # print("Change %s to %s" % (labels[idx], new_label))
                labels[idx] = new_label
                normed_labels[idx] = new_label
                change = change + 1
            idx = end_idx + 1
        elif labels[idx].startswith("S-"):
            idx = idx + 1
        elif labels[idx].startswith("M-"):
            new_label = "B-" + labels[idx][2:]
            # print("Change %s to %s" % (labels[idx], new_label))
            normed_labels[idx] = new_label
            labels[idx] = new_label
            change = change + 1
        else:
            new_label = "S-" + labels[idx][2:]
            # print("Change %s to %s" % (labels[idx], new_label))
            normed_labels[idx] = new_label
            labels[idx] = new_label
            change = change + 1

    return normed_labels, change


def evalInstance(gold, predict):
    glength, plength = gold.length, predict.length
    if glength != plength:
        raise Exception('gold length does not match predict length.')

    gold_entity_num, predict_entity_num, correct_entity_num = 0, 0, 0
    goldlabels, predictlabels = gold.labels, predict.labels

    if gold.span:
        gold_entities = label_to_entity(goldlabels)
        predict_entities = label_to_entity(predictlabels)
        gold_entity_num, predict_entity_num = len(gold_entities), len(predict_entities)
        for one_entity in gold_entities:
            if one_entity in predict_entities:
                correct_entity_num = correct_entity_num + 1
    else:
        gold_entity_num, predict_entity_num = len(goldlabels), len(predictlabels)
        for idx in range(glength):
            if goldlabels[idx] == predictlabels[idx]:
                correct_entity_num = correct_entity_num + 1

    return gold_entity_num, predict_entity_num, correct_entity_num


def readInstance(file):
    min_count = 1
    total = 0
    words = []
    for line in file:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '' or line.strip().startswith('#'):
            if len(words) > min_count:
                total += 1
                yield Sentence(words)
            words = []
        elif len(tok) == 3:
            try:
                words.append(Word(int(tok[0]), tok[1], tok[2]))
            except Exception:
                pass
        else:
            pass

    if len(words) > min_count:
        total += 1
        yield Sentence(words)

    print("Total num: ", total)


def writeInstance(filename, sentences):
    with open(filename, 'w') as file:
        for sentence in sentences:
            for entry in sentence.words:
                file.write(str(entry) + '\n')
            file.write('\n')


def printInstance(output, sentence):
    for entry in sentence.words:
        output.write(str(entry) + '\n')
    output.write('\n')