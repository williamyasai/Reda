
import math as math 
import os
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

data_index = 0

def getOneHot(text):
    
    text = pd.DataFrame(text)
    text.rename(columns={0:"Word"},inplace = True)

    le = preprocessing.LabelEncoder()
    trans = text.apply(le.fit_transform)

    enc = preprocessing.OneHotEncoder()
    enc.fit(trans)
    onehot = enc.transform(trans).toarray()
    return onehot

def parseTwittercsv():
    os.chdir('../Reda_Data')
    f = open('twcs.csv')
    csv_reader = csv.reader(f, delimiter = ',')
    responses = []
    for line in csv_reader:
        responses.append(line[4])
    return responeses

def parse():
    os.chdir('../Reda_Data')
    f = pd.read_csv('twcs.csv')
    f = f['text']
    vocab = []
    sentences = []
    i = 0
    for index,row in f.to_frame().iterrows():
        words = row['text'].split()
        sentences.append(row['text'].split())
        for word in words:
            if '@' not in word:
                vocab.append(word)
        i += 1

        if(i==800000):
            break
    return vocab,sentences


def main():
    vocab, sentences = parse()
    onehot = getOneHot(vocab)

