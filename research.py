
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
    vocab = set()
    sentences = []
    i = 0
    for index,row in f.to_frame().iterrows():
        words = row['text'].split()
        sentences.append(row['text'].split())
        for word in words:
            if '@' not in word:
                vocab.add(word)
        i += 1
        if i == 1000:
            break
    return list(vocab),sentences

def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels



def cbow_tf(vocab,onehot,sentences):
    # define size
    vocabulary_size = len(vocab)
    embedding_size = 32

    #create embeddings
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], -1.0, 1.0))

    # weigths and biases for negative sampling (look up nce regression on goggle)
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # placeholder
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))
    # We use the SGD optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    for inputs, labels in generate_batch(batch_size,num_skips,skip_window):
        feed_dict = {train_inputs: inputs, train_labels: labels}
        cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

def main():
    vocab, sentences = parse()
    onehot = getOneHot(vocab)
    cbow_tf(vocab,onehot,sentences)
main()
