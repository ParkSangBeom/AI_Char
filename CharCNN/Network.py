import tensorflow as tf

class Network:
    _learning_rate = 0.05
    _dropout_rate = 0.7

    _character_size = 250
    _embedding = 10

    _input_size = 400

    def __init__(self, sess : tf.Session, name : str):
        self._sess = sess
        self._name = name

        self._BuildNetwork()
        self._sess.run(tf.global_variables_initializer())

    def _BuildNetwork(self):
        self.X = tf.placeholder(dtype = tf.int32, shape = [None, self._input_size])
        self.Y = tf.placeholder(dtype = tf.int32, shape = [None, 1])
        one_hot_y = tf.reshape(tf.one_hot(self.Y, 2), [-1, 2])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        char_embedding = tf.get_variable(name="char_embedding", shape=[self._character_size, self._embedding])
        self.A = char_embedding
        embedded = tf.nn.embedding_lookup(char_embedding, self.X)
        self.B = embedded
        embedded = tf.reshape(embedded, [-1, 20, 20, self._embedding])

        layout_list = []
        for i in range(2):
            f_size = 3
            layout_list.append(self.GetLayout(embedded, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 6
            layout_list.append(self.GetLayout(embedded, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 9
            layout_list.append(self.GetLayout(embedded, f_size, str(f_size) + "_" + str(i)))

        L = tf.concat(layout_list, 3)
        L = tf.reshape(L , [-1, 6])
        
        W1 = tf.get_variable("W1", shape=[6, 2], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([2]))
        L1 = tf.matmul(L, W1) + b1


        loss_i = tf.nn.softmax_cross_entropy_with_logits(logits = L1, labels = one_hot_y)
        self.loss = tf.reduce_mean(loss_i)
        self.train = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(L1), 1), tf.argmax(one_hot_y, 1))
        self.accuracy = tf.cast(correct_prediction, tf.float32)

    def GetLayout(self, embedded, filter_size, name):
        conv_name = "Conv_" + str(name)
        with tf.name_scope(conv_name):
            W = tf.get_variable(name, shape = [20, filter_size, self._embedding, 1])
            b = tf.Variable(tf.random_normal([1]))
            L = tf.nn.conv2d(embedded, W, strides=[1, 1, 1, 1], padding='VALID') + b
            L = tf.nn.relu(L)
            L = tf.nn.max_pool(L, ksize=[1, 1, L.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            L = tf.nn.dropout(L, keep_prob = self.dropout_keep_prob)
            return L

    def Train(self, datas, labels):
        _, loss, A, B = self._sess.run([self.train, self.loss, self.A, self.B], feed_dict={self.X : datas, self.Y : labels, self.dropout_keep_prob : self._dropout_rate})

    def Accuracy(self, datas, labels):
        accuracy = self._sess.run([self.accuracy], feed_dict={self.X : datas, self.Y : labels, self.dropout_keep_prob : 1.0})
        return accuracy
