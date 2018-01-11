#!/usr/local/bin/ python3
# -*- coding:utf-8 -*-
# __author__ = "zenmeder"

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split

data = pd.read_csv('seattleWeather_1948-2017.csv')
def xday(s):
    s = datetime.datetime.strptime(s,"%Y-%m-%d")
    return s.timetuple().tm_yday
data['DATE'] = data['DATE'].apply(xday)
data.dropna(inplace=True)
X = data.iloc[:,[1,2,3,4]]
y = data['RAIN'].apply(lambda x: int(x))
X_train, X_test, y_train, y_test = train_test_split(X,y)

x = tf.placeholder("float", [None, 4])
W = tf.Variable(tf.zeros([4,2]))
b = tf.Variable(tf.zeros([2]))
a = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,2])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = data.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
