# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:57:09 2019

@author: GUR45397
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:51:15 2019

@author: GUR45397
"""
import tensorflow as tf
import pandas as pd
from skimage import io
from skimage.transform import resize
import numpy as np
import pickle
from keras.models import load_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping  
from keras.models import Model
from tensorflow.python import debug as tf_debug
import time
tf.compat.v1.reset_default_graph()

path = "D:/repo/autoencoder/UdacityDataset/dataset/alldata/"
imageList = []
angleList = []

d_data = {}
def predictionFromPhase1():
    try:
        df = pd.read_csv(path+"out.csv")
        df = df[df['frame_id'] == 'center_camera']
        i = 0
        
        for index, row in df.iterrows():
            filename = row['filename']
            angle = row['angle']
            image = io.imread(path+filename)
            r_image = resize(image,(66,200))
            
            imageList.append(r_image)
            angleList.append(angle)
            i = i + 1
            if(i%500 == 0):
               print(i) 
               break
    except Exception as e:
        print("***********************:",e)
        

predictionFromPhase1()

imageArray = np.asarray(imageList)
angleArray = np.asarray(angleList)
angleArray=angleArray.reshape(angleArray.shape[0],1)


tf.compat.v1.disable_eager_execution()
height = 66
width = 200
channels = 3
tf.compat.v1.reset_default_graph()



inputVal = tf.compat.v1.placeholder(tf.float32, shape = [None ,height,width,channels],name = 'features')
output = tf.compat.v1.placeholder(tf.float32, shape = [None,1],name = 'output')
lr = tf.compat.v1.placeholder(tf.float32, name = 'lr')

filter1 = tf.compat.v1.get_variable('weights1', [5, 5, 3, 24], 
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
                              dtype=tf.float32)    
conv1 = tf.nn.conv2d(inputVal, filter1, strides = 2, padding='VALID', name = 'conv1')
conv1 = tf.nn.relu(conv1)

filter2 = tf.compat.v1.get_variable('weights2', [5, 5, 24, 36], 
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
                              dtype=tf.float32)    
conv2 = tf.nn.conv2d(conv1, filter2, strides = 2, padding='VALID', name = 'conv2')
conv2 = tf.nn.relu(conv2)

filter3 = tf.compat.v1.get_variable('weights3', [5, 5, 36, 48], 
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
                              dtype=tf.float32)    
conv3 = tf.nn.conv2d(conv2, filter3, strides = 2, padding='VALID', name = 'conv3')
conv3 = tf.nn.relu(conv3)

filter4 = tf.compat.v1.get_variable('weights4', [3, 3, 48, 64], 
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
                              dtype=tf.float32)    
conv4 = tf.nn.conv2d(conv3, filter4, strides = 1, padding='VALID', name = 'conv4')
conv4 = tf.nn.relu(conv4)

filter5 = tf.compat.v1.get_variable('weights5', [3, 3, 64, 64], 
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), 
                              dtype=tf.float32)    
conv5 = tf.nn.conv2d(conv4, filter5, strides = 1, padding='VALID', name = 'conv5')
conv5 = tf.nn.relu(conv5)

conv5_reshaped = tf.reshape(conv5,shape = (-1,conv5.shape[1]*conv5.shape[2]*conv5.shape[3]))

#dense1_1  = tf.compat.v1.layers.dense(conv5_reshaped,512,activation=tf.nn.relu)
with tf.device("/job:local/task:1"):
    dense1_cluster0  = tf.compat.v1.layers.dense(conv5_reshaped,1024,activation=tf.nn.relu)
with tf.device("/job:local/task:0"):
    dense1_cluster1  = tf.compat.v1.layers.dense(conv5_reshaped,1024,activation=tf.nn.relu)
    dense1 = tf.concat([dense1_cluster0, dense1_cluster1],1)

#dense2  = tf.compat.v1.layers.dense(dense1,256,activation=tf.nn.relu)
with tf.device("/job:local/task:1"):
    dense2_cluster0  = tf.compat.v1.layers.dense(dense1,512,activation=tf.nn.relu)
with tf.device("/job:local/task:0"):
    dense2_cluster1  = tf.compat.v1.layers.dense(dense1,512,activation=tf.nn.relu)
    dense2 = tf.concat([dense2_cluster0, dense2_cluster1],1)


#dense3  = tf.compat.v1.layers.dense(dense2,128,activation=tf.nn.relu)
with tf.device("/job:local/task:1"):
    dense3_cluster0  = tf.compat.v1.layers.dense(dense2,256,activation=tf.nn.relu)
with tf.device("/job:local/task:0"):
    dense3_cluster1  = tf.compat.v1.layers.dense(dense2,256,activation=tf.nn.relu)
    dense3 = tf.concat([dense3_cluster0, dense3_cluster1],1)


dense4  = tf.compat.v1.layers.dense(dense3,1)#,activation=tf.nn.relu)
residual_error = tf.pow(dense4-output,2)
cost = tf.reduce_mean(residual_error) 
loss_summary = tf.compat.v1.summary.scalar(name="loss",tensor=cost)
r_error = tf.compat.v1.summary.scalar(name="error",tensor=residual_error[0][0])

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(cost)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session("grpc://localhost:2222") as sess:
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "GURMUNI114487:7000")
    summ_writer = tf.compat.v1.summary.FileWriter(path+"/out",sess.graph)
    init.run()
    stime = time.time()
    
    epoch = 100
    previous = 0
    for i in range(epoch):
        '''
        r,ls, z ,loss,error,f1,f2,f3,f4,f5,d4,d3,d2,d1 = sess.run((r_error,loss_summary, train_op,cost, residual_error, filter1, filter2,filter3,filter4,filter5, dense4,dense3,dense2,dense1), 
                                feed_dict={inputVal:imageArray, 
                                      output:angleArray,
                                      lr:0.001})
        '''
        d4,d3,ls,_,loss = sess.run((dense4,dense3, loss_summary, train_op, cost), 
                                feed_dict={inputVal:imageArray, 
                                            output:angleArray,
                                            lr:0.001})
        print("***************************************************",loss, d4, i) 
        summ_writer.add_summary(ls, i)
        #summ_writer.add_summary(r, i)
        #if(previous != loss):
            #print("***************************************************",loss)    
        #    previous = loss
        #print("error:",error)
    print(time.time()-stime)    