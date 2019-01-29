
import os
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing


def getOneHot():
    text = parse()
    text = pd.DataFrame(text)
    le = preprocessing.LabelEncoder()
    text = text.to_frame()
    text = text.apply(split())
    trans = text.apply(le.fit_transform)
    print(trans.head(4))
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
    vocab = set()
    i = 0
    for index,row in f.to_frame().iterrows():
        words = row['text'].split()
        for word in words:
            if '@' not in word:
                vocab.add(word)
        i += 1
        if i == 10000:
            break
    return vocab


getOneHot()
