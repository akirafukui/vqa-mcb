import numpy as np
from visual_genome_python_driver.src.local import *
import re
import pdb
import json


# True: Augments the genome dataset by converting literal number answers to numerical form (i.e. 'one' --> '1')
# For the challenge submission, we set this as False
AUGMENT = False

# Path to genome data
DATA_PATH = 'genome'

# List of words to prune
ELIMINATE = ['on', 'the', 'a', 'in', 'inside', 'at', 'it', 'is', 'with', 'near', 'behind', 
             'front', 'of', 'next', 'to']


QAs = GetAllQAs(DATA_PATH)
image_data = GetAllImageData(DATA_PATH)

def tokenize(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.strip('.')
    return [i for i in re.split(r"([-.\"',.:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i != '' and i != ' ' and i != '\n']


def remove_words(tokens, eliminate):
    pruned = []
    for token in tokens:
        if token not in eliminate:
            pruned.append(token)
    return pruned

################ Create Dictionary for text2int ######################
numwords = {}
units = [ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen" ]
tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
scales = ["hundred", "thousand", "million", "billion", "trillion"]
numbers = units + tens + scales

numwords["and"] = (1, 0)
for idx, word in enumerate(units):
    numwords[word] = (1, idx)
for idx, word in enumerate(tens):
    numwords[word] = (1, idx * 10)
for idx, word in enumerate(scales):
    numwords[word] = (10 ** (idx * 3 or 2), 0)

#######################################################################

def text2int(textnum):
        if textnum == 'hundred':
            return unicode(str(100))
        if textnum == 'thousand':
            return unicode(str(1000))

        current = result = 0
        for word in textnum.split():
            if word not in numwords:
                return textnum
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        return unicode(str(result + current))


def preprocess(aug=True):
    processed_QAs = []

    for qa_pairs in QAs:
        for qa_pair in qa_pairs:
            tokens = tokenize(qa_pair.answer)
            answer = remove_words(tokens, ELIMINATE)
            if len(answer) == 1:
                if aug and (answer[0] in numbers):
                    final_answer = text2int(answer[0])
                else:
                    final_answer = answer[0]
                entry = {'id': qa_pair.id, 'image': qa_pair.image.id,
                         'question': qa_pair.question, 'answer': final_answer}
                processed_QAs.append(entry)
    return processed_QAs


data = preprocess(AUGMENT)
if AUGMENT:
    with open('genome/question_answers_prepro_aug.json', 'w') as f:
        json.dump(data, f)
else:
    with open('genome/question_answers_prepro.json', 'w') as f:
        json.dump(data, f)

