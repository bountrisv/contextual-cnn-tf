import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, kmax-pooling, convolutional, maxpooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Variables from paper
        assumed_value_K = 15
        num_filters_2 = num_filters
        filter_size_2 = 4

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            W_emb = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W_emb")
            self.embedded_chars = tf.nn.embedding_lookup(W_emb, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + kmaxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-kmaxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")

                # Main addition to the original implementation is adding K max pooling
                # Optimally we would add a custom op for this
                t = tf.transpose(h, perm=[0, 3, 2, 1])
                d = tf.shape(t)[-1]
                _, indices = tf.nn.top_k(t, k=assumed_value_K, sorted=False, name="k_max")
                # Create one - hot boolean tensor
                one_hot = tf.one_hot(indices, d, on_value=True, off_value=False, dtype=tf.bool)
                # Reduce it to shape t
                temporary = tf.reduce_sum(tf.cast(one_hot, tf.float32), 3)
                result_flat = tf.boolean_mask(t, tf.not_equal(tf.cast(temporary, tf.bool), False))
                result = tf.reshape(result_flat, tf.shape(indices))
                kmax_pooled = tf.transpose(result, perm=[0, 3, 2, 1])
                pooled_outputs.append(kmax_pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(3, pooled_outputs)

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h1_drop = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)

        # Convolutional layer capturing sequential features
        with tf.name_scope("conv-maxpool"):
            num_filters_total = num_filters * len(filter_sizes)
            filter_shape = [filter_size_2, 1, num_filters_total, num_filters_2]
            W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b2")
            conv2 = tf.nn.conv2d(
                self.h1_drop,
                W2,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv2')
            self.h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu")

            max_pool = tf.nn.max_pool(
                self.h2,
                ksize=[1, assumed_value_K - filter_size_2 + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='max-pool')

        # Add dropout
        with tf.name_scope("dropout2"):
            self.h2_drop = tf.nn.dropout(max_pool, self.dropout_keep_prob)

        # Add last layer
        self.h2_flat = tf.reshape(self.h2_drop, [-1, num_filters_2])
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_2, num_classes],  # to fix
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W_emb)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h2_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
