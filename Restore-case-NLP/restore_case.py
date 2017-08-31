#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 19:38:13 2017

@author: francescobattocchio
"""
import sys
import nltk
import codecs
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

# Load previously trained model
clf = joblib.load('classifier.pkl')

# Extract features for the model 
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index][0],
        'is_first': index == 0,
        'is_punctuated': sentence[index][0] != '.' and sentence[index][0][-1] == '.',
        'prev_word': 'None' if index == 0 else sentence[index - 1][0],
        'next_word': 'None' if index == len(sentence) - 1 else sentence[index + 1][0],
        'has_hyphen': '-' in sentence[index][0],
        'is_numeric': sentence[index][0].isdigit(),
        'POS': sentence[index][1],
        'prev_POS': 'None' if index == 0 else sentence[index - 1][1],
        'next_POS': 'None' if index == len(sentence) - 1 else sentence[index + 1][1],
    }

# Build sentence from tokens
import string
def detokenize(tokenized):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokenized]).strip()

# Model function: recover casing in upper case sentence
def correct_capital(sentence):
    # convert to lower case and POS tag
    sentence = nltk.pos_tag([word.lower() for word in nltk.word_tokenize(sentence)])
    new_sentence = []
    for index_1 in range(len(sentence)):
        if clf.predict(features(sentence, index_1))==True:
            word = sentence[index_1][0].title()
        else:
            word = sentence[index_1][0]
        new_sentence.append(word)
    return detokenize(new_sentence)

# Read text from file
def read_file(file_name):
#    f = open(file_name,"r") #opens file with name of "test.txt"
    f = codecs.open(file_name, "r", "utf-8")
    text = f.read()
#    print(text)
    f.close()
    return text

# Correct upper case text separated by new lines
def correct_text(text_file):
    text = read_file(text_file)
    sentences = text.split('\n') # assume that sentences are separated by new lines
    sentences_capitalised = [correct_capital(sentence) for sentence in sentences]
    text_capitalised = ' \n '.join([sentence for sentence in sentences_capitalised]) 
    print(text_capitalised)

if __name__ == "__main__":
    a = str(sys.argv[1])
    correct_text(a)