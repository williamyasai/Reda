
import skipgram
import research
import collections
import sys
import csv
import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')

    plt.savefig(filename)

vocab, sentences = research.parse()

batch_size = 64
embedding_size = 256
skip_window = 4
num_skips = 1
negative_sampled = 16


'''
network = skipgram.skipgram(embedding_size,batch_size,negative_sampled,num_skips,skip_window,10000)
network.organize_data(vocab)
batch,labels = network.generate_batch()
final_embeddings = network.run()
reverse_dictionary = network.reversed_dictionary
w = csv.writer(open("output.csv", "w"))
for key, val in reverse_dictionary.items():
    w.writerow([key, val])
'''


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    for p in [2,5,6,7,8,9] :
        tsne = TSNE(perplexity=p, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(1,plot_only)]
        plot_with_labels(low_dim_embs, labels, '../Reda/Visualizations/tsne{}.png'.format(p))

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)

