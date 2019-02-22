
import skipgram
import research
import collections

vocab, sentences = research.parse()


batch_size = 64
embedding_size = 64
skip_window = 1
num_skips = 2
negative_sampled = 16

network = skipgram.skipgram(embedding_size,batch_size,negative_sampled,num_skips,skip_window,10000)
network.organize_data(vocab)
batch,labels = network.generate_batch()
network.run()

