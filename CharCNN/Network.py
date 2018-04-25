import tensorflow as tf
import TensorBoard as tb
import Saver as sv

class Network:
    _learning_rate = 1e-3
    #_learning_rate = 0.00001
    _dropout_rate = 0.7

    _character_size = 250
    _embedding = 40

    _input_size = 400

    def __init__(self, sess : tf.Session, name : str):
        self._sess = sess
        self._name = name
        
        self._BuildNetwork()   
        self._sess.run(tf.global_variables_initializer())

        self._tb = tb.TensorBoard(name, sess)
        self._SetTensorBoard()

        self._saver = sv.Saver(name, sess)
        self._SetSaver()

    def _BuildNetwork(self):
        self.X = tf.placeholder(dtype = tf.int32, shape = [None, self._input_size])
        self.Y = tf.placeholder(dtype = tf.int32, shape = [None, 1])
        one_hot_y = tf.reshape(tf.one_hot(self.Y, 2), [-1, 2])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        char_embedding = tf.get_variable(name="char_embedding", shape=[self._character_size, self._embedding])
        embedded = tf.nn.embedding_lookup(char_embedding, self.X)
        #embedded = tf.reshape(embedded, [-1, 2, 200, self._embedding])

        layout_list = []

        embedded2 = tf.reshape(embedded, [-1, 2, 200, self._embedding])
        embedded4 = tf.reshape(embedded, [-1, 4, 100, self._embedding])
        embedded10 = tf.reshape(embedded, [-1, 10, 40, self._embedding])

        for i in range(1):
            f_size = 2
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 3
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 4
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 5
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 6
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 7
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 8
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 9
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))

        for i in range(1):
            f_size = 10
            layout_list.extend(self.GetLayout(embedded2, f_size, 2, str(f_size) + "_" + str(i) + "_" + str(2)))


        for i in range(1):
            f_size = 2
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 3
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 4
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 5
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 6
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 7
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 8
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))

        for i in range(1):
            f_size = 9
            layout_list.extend(self.GetLayout(embedded4, f_size, 4, str(f_size) + "_" + str(i) + "_" + str(4)))


        for i in range(1):
            f_size = 2
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 3
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 4
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 5
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 6
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 7
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 8
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 9
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        for i in range(1):
            f_size = 10
            layout_list.extend(self.GetLayout(embedded10, f_size, 10, str(f_size) + "_" + str(i) + "_" + str(10)))

        lenth = len(layout_list)
        hidden = int(lenth / 2)
        L = tf.concat(layout_list, 3)
        L = tf.reshape(L , [-1, lenth])
        
        W1 = tf.get_variable("W1", shape=[lenth, hidden], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([hidden]))
        L1 = tf.matmul(L, W1) + b1

        W2 = tf.get_variable("W2", shape=[hidden, 2], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([2]))
        L2 = tf.matmul(L1, W2) + b2

        loss_i = tf.nn.softmax_cross_entropy_with_logits(logits = L2, labels = one_hot_y)
        self.loss = tf.reduce_mean(loss_i)
        self.train = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(L2), 1), tf.argmax(one_hot_y, 1))
        self.accuracy = tf.cast(correct_prediction, tf.float32)

    def GetLayout(self, embedded, filter_size, width, name):
        conv_name = "Conv_" + str(name)
        with tf.name_scope(conv_name):
            #W = tf.get_variable(name, shape = [20, filter_size, self._embedding, 1])
            #b = tf.Variable(tf.random_normal([1]))
            #L = tf.nn.conv2d(embedded, W, strides=[1, 1, 1, 1], padding='VALID') + b
            #L = tf.nn.relu(L)
            #L = tf.nn.max_pool(L, ksize=[1, 1, L.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            #L = tf.nn.dropout(L, keep_prob = self.dropout_keep_prob)
            #return L

            #W1 = tf.get_variable(name + str("a"), shape = [3, 3, self._embedding, 10])
            #b1 = tf.Variable(tf.random_normal([1]))
            #L1 = tf.nn.conv2d(embedded, W1, strides=[1, 1, 1, 1], padding='VALID') + b1
            #L1 = tf.nn.relu(L1)
            ##print(L1, "######")
            #L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

            #print(L1, "@@@@@@@@@@@@@@@@@@")

            W2 = tf.get_variable(name + str("b"), shape = [width, filter_size, self._embedding, 64])
            b2 = tf.Variable(tf.random_normal([1]))
            L2 = tf.nn.conv2d(embedded, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 1, L2.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')

            LL = []
            split = tf.split(L2, 64, 3)
            for s in split:
                M = tf.nn.max_pool(s, ksize=[1, 1, s.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
                M = tf.nn.dropout(M, keep_prob = self.dropout_keep_prob)
                LL.append(M)

            return LL

    def _SetTensorBoard(self):
        self._tb.Scalar("Loss_Value", self.loss)
        self._tb.Merge()

    def _WirteTensorBoard(self, summary, step):
        self._tb.writer.add_summary(summary, global_step = step)

    def _SetSaver(self):
        self._saver.CheckRestore()

    def _Save(self, step):
        if step != 0 and step % 10000 == 0:
            self._saver.Save()
            print("Save!!")

    def Train(self, datas, labels, step):
        _, loss, summary = self._sess.run([self.train, self.loss, self._tb.merge_summary], feed_dict={self.X : datas, self.Y : labels, self.dropout_keep_prob : self._dropout_rate})
        #print(loss)

        self._WirteTensorBoard(summary, step)
        self._Save(step)

    def Accuracy(self, datas, labels):
        accuracy = self._sess.run([self.accuracy], feed_dict={self.X : datas, self.Y : labels, self.dropout_keep_prob : 1.0})
        return accuracy
