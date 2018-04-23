import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from datetime import datetime
import TensorBoard as tb
import sys

class Network:
    _learning_rate = 0.001
    _epoch = 500
    _dropout_rate = 0.7

    def __init__(self, sess : tf.Session, name : str, data):
        self._sess = sess
        self._name = name
        self._data = data

        self._BuildNetwork()
        self._sess.run(tf.global_variables_initializer())

    def _BuildNetwork(self):
        self.X = tf.placeholder(dtype = tf.int32, shape = [None, 625])
        one_hot_x = tf.one_hot(self.X, 100)
        X = tf.reshape(one_hot_x, [-1, 1, 625, 100])

        self.Y = tf.placeholder(dtype = tf.int32, shape = [None, 1])
        one_hot_y = tf.one_hot(self.Y, 2)
        one_hot_y = tf.reshape(one_hot_y, [-1, 2])

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        con = []
        #for i in range(2):
        #    f_size = 1
        #    con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        #for i in range(1):
        #    f_size = 2
        #    con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        #for i in range(1):
        #    f_size = 3
        #    con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        
        for i in range(2):
            f_size = 4
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 6
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 8
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 10
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 12
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        for i in range(2):
            f_size = 14
            con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))
        #for i in range(2):
        #    f_size = 11
        #    con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))

        #for i in range(2):
        #    f_size = 16
        #    con.extend(self.GetLayout(X, f_size, str(f_size) + "_" + str(i)))
        L = tf.concat(con, 3)
        lenth = int(L.get_shape()[3])
        L_flat = tf.reshape(L , [-1, 1 * 1 * lenth])

        W10 = tf.get_variable("W10", shape=[lenth, 2], initializer=tf.contrib.layers.xavier_initializer())
        b10 = tf.Variable(tf.random_normal([2]))

        hy = tf.matmul(L_flat, W10) + b10
        loss_i = tf.nn.softmax_cross_entropy_with_logits(logits = hy, labels = one_hot_y)
        self.loss = tf.reduce_mean(loss_i)
        self.train = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(hy), 1), tf.argmax(one_hot_y, 1))
        self.accuracy = tf.cast(correct_prediction, tf.float32)

    def GetLayout(self, X, filter_size, name):
        conv_name = "Conv_" + str(name)
        with tf.name_scope(conv_name):           
            W = tf.get_variable(name, shape = [1, filter_size, 100, 1])
            b = tf.Variable(tf.random_normal([1]))
            L = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='VALID') + b
            L = tf.nn.relu(L)
            #L = tf.nn.dropout(L, keep_prob = self.dropout_keep_prob)
            #L = tf.nn.max_pool(L, ksize=[1, 1, L.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
            #print(L, "@@@@@@@@")

            LL = []
            split = tf.split(L, 2, 2)
            for s in split:
                M = tf.nn.max_pool(s, ksize=[1, 1, s.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
                M = tf.nn.dropout(M, keep_prob = self.dropout_keep_prob)
                LL.append(M)

            #LL = tf.nn.dropout(LL, keep_prob = self.dropout_keep_prob)
            return LL

    def Train(self, opinions, scores):
        _, loss = self._sess.run([self.train, self.loss], feed_dict={self.X : opinions, self.Y : scores, self.dropout_keep_prob : self._dropout_rate})
        print(loss)

    def Accuracy(self, opinions, scores):
        accuracy = self._sess.run([self.accuracy], feed_dict={self.X : opinions, self.Y : scores, self.dropout_keep_prob : 1.0})
        return accuracy

    #def __init__(self, sess : tf.Session, name : str, data):
    #    self._sess = sess
    #    self._name = name
    #    self._tb = tb.TensorBoard(name, sess)

    #    self._ConvertData(data)
    #    self._BuildNetwork()
    #    self._SetTensorBoard()

    #def _BuildNetwork(self):    
    #    self.X = tf.placeholder(tf.int32, [None, self._sequence_length])  # X data
    #    self.Y = tf.placeholder(tf.int32, [None, self._sequence_length])  # Y label
    #    self.A = x_one_hot = tf.one_hot(self.X, self._num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
    #    self.B = y_one_hot = tf.one_hot(self.Y, self._num_classes)
    #    #cell = tf.contrib.rnn.BasicLSTMCell(num_units = self._hidden_size, state_is_tuple = True)   
    #    #multi_cells = rnn.MultiRNNCell([self.CreateCell() for _ in range(3)], state_is_tuple=True)
    #    #outputs, states = tf.nn.dynamic_rnn(multi_cells, x_one_hot, dtype=tf.float32)

    #    enc_cell = rnn.MultiRNNCell([self.CreateCell() for _ in range(2)], state_is_tuple=True)
    #    dec_cell = rnn.MultiRNNCell([self.CreateCell() for _ in range(2)], state_is_tuple=True)      

    #    with tf.variable_scope('encode'):
    #        outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, x_one_hot, dtype=tf.float32)

    #    with tf.variable_scope('decode'):
    #        outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, x_one_hot, dtype=tf.float32, initial_state=enc_states)

    #    # FC layer
    #    X_for_fc = tf.reshape(outputs, [-1, self._hidden_size])
    #    outputs = tf.contrib.layers.fully_connected(X_for_fc, self._num_classes, activation_fn = None)

    #    # reshape out for sequence_loss
    #    outputs = tf.reshape(outputs, [-1, self._sequence_length, self._num_classes])
    #    weights = tf.ones([self._batch_size, self._sequence_length])

    #    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = self.Y, weights = weights)
    #    self._loss = tf.reduce_mean(sequence_loss)
    #    self._train = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

    #    self.prediction = tf.argmax(outputs, axis=2) 
    #    self.A = outputs

    #def CreateCell(self):
    #    cell = rnn.BasicLSTMCell(num_units = self._hidden_size, state_is_tuple=True)
    #    return cell

    #def _SetTensorBoard(self):
    #    self._tb.Scalar("Loss_Value", self._loss)
    #    self._tb.Merge()

    #def _ConvertData(self, data):
    #    #print(self.input_data)
    #    #print(self.output_data)
    #    #print(data._index_to_char)
    #    #print(data._char_to_index)
    #    #print(data.input_max)
    #    #print(data.output_max)

    #    #self._idx2char = list(set(data))  # index -> char
    #    #self._char2idx = {c : i for i, c in enumerate(self._idx2char)}  # char -> idex

    #    self._batch_size = len(data.input_data)  # one sample data, one batch
    #    self._sequence_length = len(max(data.input_data, key=len))  # number of lstm rollings (unit #)
    #    self._dic_size = data.index_size # RNN input size (one hot size)

    #    self._hidden_size = data.index_size  # RNN output size
    #    self._num_classes = data.index_size  # final output size (RNN or softmax, etc.)

    #    self._x_data  = data.input_data
    #    self._y_data  = data.output_data

    #def Train(self, data):
    #    for i in range(self._epoch):
    #        l, _, summary, A, B = self._sess.run([self._loss, self._train, self._tb.merge_summary, self.A, self.B], feed_dict={self.X: self._x_data, self.Y: self._y_data})

    #        result = self._sess.run(self.prediction, feed_dict={self.X: self._x_data})
    #        self._tb.writer.add_summary(summary, global_step = i)

    #        # print char using dic
    #        for c in result:
    #            result_str = [data.index_to_char[k] for k in np.squeeze(c)]
    #            print(i, "loss:", l, "Prediction:", ''.join(result_str))

    #        print("===============================")

    #    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #    while True:
    #        sys.stdout.write("> ")
    #        sys.stdout.flush()
    #        line = sys.stdin.readline()
    #        line = line.replace("\n", "")
    #        #for aa in line:
    #        #    if aa == "\n":
    #        #        continue
    #        #    bb.extend(aa)

    #        bb = data.StrToOnehot([line])


    #        result = self._sess.run(self.prediction, feed_dict={self.X: bb})
    #        for c in result:
    #            result_str = [data.index_to_char[k] for k in np.squeeze(c)]
    #            print(''.join(result_str))
    #        #print(self._get_replay(line.strip()))

    #        #sys.stdout.write("\n> ")
    #        #sys.stdout.flush()

    #        #line = sys.stdin.readline()

    #    #aa = self._sess.run(self.prediction, feed_dict={self.X: self._x_data})
    #    #for c in aa:
    #    #    result_str = [data.index_to_char[k] for k in np.squeeze(c)]
    #    #    print(''.join(result_str))

    #    #print("===========================")
    #    #bb = self._sess.run(self.prediction, feed_dict={self.X: self._x_data})
    #    #for c in bb:
    #    #    result_str = [data.index_to_char[k] for k in np.squeeze(c)]
    #    #    print(''.join(result_str))