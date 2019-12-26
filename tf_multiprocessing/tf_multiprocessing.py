# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:16:51 2019

@author: GUR45397
"""

import numpy as np
import tensorflow as tf
import time

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

x = tf.placeholder(tf.float32, 10)


with tf.device("/job:local/task:1"):
    first_batch = tf.slice(x, [0], [5])
    mean1 = tf.reduce_mean(first_batch)

with tf.device("/job:local/task:0"):
    second_batch = tf.slice(x, [5], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2


with tf.Session("grpc://localhost:2222") as sess:
    starttime = time.time()
    for i in range(10000):
        f1,s2,m,m1,m2 = sess.run((first_batch, second_batch, mean,mean1,mean2), feed_dict={x: np.array([1,2,3,4,5,6,7,8,9,10])})
		
        print("f1:",f1)
        print("s2:",s2)
        print("m:",m," m1:",m1," m2:",m2)
    print(time.time()-starttime)