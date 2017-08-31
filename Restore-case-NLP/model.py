#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 11:04:43 2017

@author: francescobattocchio
"""
import nltk
import codecs
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import pprint
 
# Import corpus
# tagged_sentences = nltk.corpus.brown.tagged_sents()
# Extract sentences form corpus
sentences = nltk.corpus.treebank.sents()

# Corpus converted into lower case and POS tag
sentences_lower = [[word.lower() for word in sentence] for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in sentences_lower]
 
print(sentences[0])
print(tagged_sentences[0])
print ("Tagged sentences: ", len(tagged_sentences)) # 3914
print ("Tagged words: ", nltk.corpus.treebank.words()) # 100676
 
# Extract features to train the model (need to be improved)
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
     
pprint.pprint(features(nltk.pos_tag(['This', 'is', 'a',',', 'sentence','!']), 3))

# Tranform data set to train and test the model:
# X : we will predict on all lower case text, thus the feaures need to be extracted on lower case text
# y : labels are extracted on mixed case text: True if word is capitalised or False if word is not capitalised
def transform_to_dataset(sentences, tagged_sentences):
    X, y = [], []
    # label according to capital or not capital (exclude non alphabetic)
    for sentence in sentences:
        for index in range(len(sentence)):
            label = sentence[index][0].isalpha() and sentence[index][0][0].upper() == sentence[index][0][0]
            y.append(label)
    # extract features in lower case tagged sentences
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(tagged, index))
    return X, y

# Split the dataset for training and testing
cutoff = int(.75 * len(sentences))
X_train, y_train = transform_to_dataset(sentences[:cutoff], tagged_sentences[:cutoff])
X_test, y_test = transform_to_dataset(sentences[cutoff:], tagged_sentences[cutoff:])
 
print(cutoff)         # 2935
print(len(X_train))   # 75784
print(len(X_test))    # 24892

# Decision tree classifier (need to test other classifiers and optimise parameters)
 
clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])
 
# Train model
clf.fit(X_train[:20000], y_train[:20000]) # train in only 20k samples to save some time...  
 
print('Training completed')

# Evaluate model
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred.tolist()))

############### Model testing   #############################  
# Define some functions

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

# Example of use of correct_capital in sentences from the testing set
#index_random = np.random.randint(low=cutoff, high=len(sentences))
#sent = sentences[index_random]
#sent_upper = [word.upper() for word in sent]
#sent_corrected = correct_capital(detokenize(sent_upper))
#print(sent_upper, '\n', sent_corrected)

# Write text to file
def write_file(file_name, text):
    f = open(file_name, 'w')
    f.write(text)
    f.close()
    
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
    return text_capitalised

# Example of use on upper case text: 
# select sentences from test set with a good variety of features and save text
#index_text = [3057, 3694, 3662]
#text = '\n'.join([detokenize(sentences[index]) for index in index_text])
#text_upper = '\n'.join([detokenize([word.upper() for word in sentences[index]]) for index in index_text])    
#write_file('Text_mixed.txt', text)
#write_file('Text_upper.txt', text_upper)
# Convert uppercase text and compare with original text
text_original = read_file('Text_mixed.txt')
text_corrected = correct_text('Text_upper.txt')
print('ORIGINAL TEXT:' , '\n', text_original, '\n', 'CAPITALs RESTORED', '\n', text_corrected)

########################   Export classifier   ###############################
## Save model to load in external file
#filename = 'classifier.pkl'
#joblib.dump(clf, filename)
##Load and test
#joblib_clf = joblib.load(filename)
#print(X_test[0]['word'])
#print(joblib_clf.predict(X_test[0]))



    