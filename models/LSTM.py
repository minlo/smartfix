import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


class LSTM_model(BaseEstimator, TransformerMixin):
    """
    This model is for LSTM, we do train and predict in this class
    furthermore, we establish another pipiline for LSTM apart from the pipeline of ML method
    """
    def __init__(self, look_forward_days=1, seq_length=30, input_dim=20,
                 output_dim=1, num_hidden=128, num_fully=256, learning_rate=0.001,
                 batch_size=100, training_steps=100, display_step=10, separate_date = pd.datetime(2017, 12, 1)):
        """
        The grid search parameters are
        :param look_forward_days:
        :param seq_length:
        :param input_dim:
        :param output_dim:
        :param num_hidden:
        :param num_fully:
        :param learning_rate:
        :param batch_size:
        :param training_steps:
        :param display_step:
        :param separate_date:
        """

        self.seq_length = seq_length
        self.look_forward_days = look_forward_days
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.num_fully = num_fully
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.display_step = display_step
        self.separate_date = separate_date

    def separate_data_by_date(self, data):
        train_size = data.iloc[data.index < self.separate_date, :].shape[0]
        test_size = data.iloc[data.index >= self.separate_date, :].shape[0]
        return train_size, test_size

    def data_separate(self, data):
        datax = []
        datay = []

        x = data
        y = data[:, [-1]]

        for i in range(self.seq_length, len(y)-self.look_forward_days):
            _x = x[i-self.seq_length:i, :]
            _y = y[i + day]
            datax.append(_x)
            datay.append(_y)

        train_size, test_size = seperate_data_bydate(data)
        datax, datay = np.array(datax), np.array(datay)

        trainx, testx = np.array(datax[0:train_size]), np.array(datax[train_size:len(datax)])
        trainy, testy = np.array(datay[0:train_size]), np.array(datay[train_size:len(datay)])
        return trainx, trainy, testx, testy

    def model_train_test(self, train_x, train_y, test_x, test_y):
        """

        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :return: the accuracy on test set
        """

        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, self.seq_length, self.input_dim], name='input_x')
        y = tf.placeholder(tf.float32, [None, self.output_dim], name='input_y')

        # weights = {
        #     'out': tf.Variable(tf.truncated_normal([self.num_hidden, 1])),
        #     'fully': tf.Variable(tf.truncated_normal([self.num_fully, 1]))
        # }
        # biases = {
        #     'out': tf.Variable(tf.truncated_normal([1])),
        #     'fully': tf.Variable(tf.truncated_normal([1]))
        # }

        weights = tf.Variable(tf.random_normal([self.n_hidden, self.output_dim]), name="weights")
        biases = tf.Variable(tf.random_normal([self.output_dim]), name="biases")

        x = tf.unstack(x, self.seq_length, 1)
        cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        outputs, _ = rnn.static_rnn(cell, x, dtype=tf.float32)
        y_pred = tf.add(tf.matmul(outputs[-1], weights), biases)
        tf.add_to_collection('predict', y_pred)
        # hidden=tf.nn.tanh(hidden)
        # result=tf.add(tf.matmul(hidden,weights['fully']),biases['fully'])

        loss = tf.reduce_sum(tf.square(y_pred - y))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        correct = (tf.abs(tf.abs(y_pred, y)) < 0.1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            num_of_batches = int(train_x.shape[0] / self.batch_size)
            for i in range(self.training_steps):
                for j in range(num_of_batches):
                    indices = random.sample(range(train_x.shape[0]), batch_size),
                    batch_x, batch_y = train_x[indices], train_y[indices]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if (i + 1) % self.display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    acc_valid = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
                    print("Step %d, training accuracy:%.2f,testing accuracy:%.2f  " % (i + 1, acc_train, acc_valid))

            acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
            print("After training and validation,testing accuracy:%.2f" % acc_test)
            saver = tf.train.Saver()
            model_path = './T+' + str(self.look_forward_days) + '_model'
            saver.save(sess, model_path)
        return acc_test

    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data_value = data.values
        trainx, trainy, testx, testy = self.data_separate(data_value)
        acc_test = self.model_train_test(trainx, trainy, testx, testy)
        return acc_test

    def predict(self, X):
        """

        :param X:
        :return: y_pred
        """
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./T+' + str(self.look_forward_days) + '_model.meta')
            new_saver.restore(sess, './T+' + str(self.look_forward_days) + '_model')
            graph = tf.get_default_graph()
            predict = tf.get_collection('predict')[0]
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            x = X.values[-1-self.seq_length:-1, :]

            res = sess.run(predict, feed_dict={input_x: x})

            return res
