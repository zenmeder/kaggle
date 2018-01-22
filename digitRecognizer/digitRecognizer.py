#!/usr/local/bin/ python3
# -*- coding:utf-8 -*-
# __author__ = "zenmeder"
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# from sklearn.model_selection import train_test_split
# data = pd.read_csv('train.csv')
# X, y = np.array(data.iloc[:,1:]),data.iloc[:,:1]
# x1, x2, y1, y2 = train_test_split(X, y, test_size=0.3, random_state=3)
# def weight_variable(shape):
#     return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
# def bias_variable(shape):
#     return tf.Variable(tf.constant(0.1, shape=shape))
# def conv2d(x,W):
#     return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')
# def max_pool(x):
#     return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# sess = tf.InteractiveSession()
# x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
# y_ = tf.placeholder(tf.float32, shape=[None, 10], name='output')
# x_ = tf.reshape(x, [-1, 28, 28, 1])
# W_conv1 = weight_variable([5,5,1,32])
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_, W_conv1)+b_conv1)
# h_pool1 = max_pool(h_conv1)
# W_conv2 = weight_variable([5,5,32,64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
# h_pool2 = max_pool(h_conv2)
# W_fc1 = weight_variable([7*7*64,1024])
# b_fc1 = bias_variable([1024])
# h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# W_fc2 = weight_variable([1024,10])
# b_fc2 = bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# tf.global_variables_initializer().run()
# import random
# def get_random_cached(x1, y1, size):
#     a, b = [],[]
#     n = len(x1)
#     for i in range(size):
#         r = random.randint(0, n-1)
#         a.append(x1[r])
#         b.append(y1[r])
#     return np.array(a), np.array(b)
#
# print("training accuracy is {0}".format(accuracy.eval(feed_dict={x:np.array(x2),y_:np.array(y2),keep_prob:1.0})))