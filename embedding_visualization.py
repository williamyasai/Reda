
import skipgram
import research
import collections
import sys
vocab, sentences = research.parse()

batch_size = 64
embedding_size = 64
skip_window = 1
num_skips = 2
negative_sampled = 16

network = skipgram.skipgram(embedding_size,batch_size,negative_sampled,num_skips,skip_window,10000)
network.organize_data(vocab)
batch,labels = network.generate_batch()
final_embeddings = network.run()
try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(),
                                                        'tsne.png'))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)

