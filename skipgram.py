import collections
import random
import tensorflow as tf
import numpy as np
import math
class skipgram:

    def __init__(self,embeddingsize = 1,batchsize = 1,negativesize = 1,num_skips = 1, skip_window = 1,vocab_size = 1):
        self.vocabulary_size = vocab_size
        self.embedding_size = embeddingsize
        self.batch_size = batchsize
        self.negative_size = negativesize
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.dataindex = 0
        self.data = []
        self.word_freq = [[]]
        self.dictionary = dict()
        self.reversed_dictionary = dict()
    def organize_data(self,words):
        n_words = self.vocabulary_size
        word_freq = [['unknown',-1]]
        word_freq.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word,_ in word_freq:
            dictionary[word] = len(dictionary)
        data = []
        unknown_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unknown_count += 1
            data.append(index)
        word_freq[0][1] = unknown_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data = data
        self.word_freq = data
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary

    def generate_batch(self):
        data_index = self.dataindex
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if data_index + span > len(self.data):
          data_index = 0
        buffer.extend(self.data[data_index:data_index + span])
        data_index += span
        for i in range(self.batch_size // self.num_skips):
          context_words = [w for w in range(span) if w != self.skip_window]
          words_to_use = random.sample(context_words, self.num_skips)
          for j, context_word in enumerate(words_to_use):
            batch[i * self.num_skips + j] = buffer[self.skip_window]
            labels[i * self.num_skips + j, 0] = buffer[context_word]
          if data_index == len(self.data):
            buffer.extend(data[0:span])
            data_index = span
          else:
            buffer.append(self.data[data_index])
            data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(self.data) - span) % len(self.data)
        self.dataindex = data_index
        return batch, labels

    def run(self):

        graph = tf.Graph()

        with graph.as_default():
            # batch size
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            # linear function mapping vector of size (V,1) to (E,1).
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            # negative sampling weighs and biases
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=self.negative_size,
                num_classes=self.vocabulary_size))

            tf.summary.scalar('loss', loss)

            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)      
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()
            
        num_steps = 100001
        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(log_dir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
              batch_inputs, batch_labels = generate_batch(self.batch_size, self.num_skips,
                                                          self.skip_window)
              feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

              # Define metadata variable.
              run_metadata = tf.RunMetadata()

              # We perform one update step by evaluating the optimizer op (including it
              # in the list of returned values for session.run()
              # Also, evaluate the merged op to get all summaries from the returned
              # "summary" variable. Feed metadata variable to session for visualizing
              # the graph in TensorBoard.
              _, summary, loss_val = session.run([optimizer, merged, loss],
                                                 feed_dict=feed_dict,
                                                 run_metadata=run_metadata)
              average_loss += loss_val

              # Add returned summaries to writer in each step.
              writer.add_summary(summary, step)
              # Add metadata to visualize the graph for the last run.
              if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

              if step % 2000 == 0:
                if step > 0:
                  average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000
                # batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

              # Note that this is expensive (~20% slowdown if computed every 500 steps)
              if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                  valid_word = reverse_dictionary[valid_examples[i]]
                  top_k = 8  # number of nearest neighbors
                  nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                  log_str = 'Nearest to %s:' % valid_word
                  for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                  print(log_str)
            final_embeddings = normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(log_dir + '/metadata.tsv', 'w') as f:
              for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

            # Save the model for checkpoints.
            saver.save(session, os.path.join(log_dir, 'model.ckpt'))

            # Create a configuration for visualizing embeddings with the labels in
            # TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
            projector.visualize_embeddings(writer, config)

            writer.close()

