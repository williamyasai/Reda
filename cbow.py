
class cbow:

    def __init__(self,windowsize = 1,vocab,embeddingsize = 1,batchsize = 1):
        self.windowsize = windowsize
        self.embeddingsize = 1
        self.batchsize = batchsize
        self.vocabulary = vocab
        self.vocabulary = vocab

    
    def run(self):

        graph = tf.Graph()

        with graph.as_default():
            # batch size
            with tf.name_scope('inputs')
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            # linear function mapping vector of size (V,1) to (E,1).
            with tf.name_scope('embeddings')
                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

            # negative sampling weighs and biases
            with tf.name_scope('weights')
                nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases')
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

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
