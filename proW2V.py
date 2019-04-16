
import multiprocessing
from gensim.models import Word2Vec



def train_word2vec(filename):
    data = gensim.models.word2vec.LineSentence(filename)
    return Word2Vec(data, size=200, window=5, min_count=5, workers=multiprocessing.cpu_count()


def main():

main()

