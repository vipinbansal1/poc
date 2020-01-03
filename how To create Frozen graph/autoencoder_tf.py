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
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
tf.compat.v1.reset_default_graph()

filename = "cfe.pk"
path = "D:/repo/autoencoder/UdacityDataset/dataset/alldata/"
'''
AE = load_model(path +"part1.hdf5")
Inp = AE.input
Outp = AE.get_layer('conv5').output
curr_layer_model = Model(Inp, Outp)
'''

'''
infile = open(path+filename,'rb')
new_dict = pickle.load(infile)
infile.close()

def get_batch(batch_size, transformation = False):
    
    listt = zip(new_dict.keys(), new_dict.values()) 
    l_obj = list(listt)
    for batch_i in range(0, len(l_obj)//batch_size):
        start_i = batch_i * batch_size
        try:
            batch = l_obj[start_i:start_i + batch_size]           
        except IndexError:
            batch = list(new_dict.items())[start_i:]
            
        cnnExtractedFeatures = []
        deltaError = []
        for i in range(0,len(batch)):
            featureData=batch[i][1]
            extractedFeatures = featureData[0:len(featureData)-2]
            actualAngle=featureData[len(featureData)-2]
            predictedAngle = featureData[len(featureData)-1]
            
            cnnExtractedFeatures.append(extractedFeatures)
            deltaError.append(abs(actualAngle-predictedAngle))
        yield cnnExtractedFeatures, deltaError

for i, (X_image,angle) in enumerate(get_batch(4, False)):
    print(i)
'''
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
            
            #predicted = curr_layer_model.predict(r_image.reshape(1,66,200,3))
            #predicted_angle = AE.predict(r_image.reshape(1,66,200,3))
            #featureRow = np.append(predicted.reshape(-1),angle)
            #featureRow = np.append(featureRow,predicted_angle[0][0])
            
            #d_data[filename.split("/")[1]] = featureRow
            
            i = i + 1
            if(i%50 == 0):
               print(i) 
               break
    except Exception as e:
        print("***********************:",e)
        

predictionFromPhase1()

o_pickle = open(filename,"wb")
pickle.dump(d_data,o_pickle)
o_pickle.close()


imageArray = np.asarray(imageList)
angleArray = np.asarray(angleList)
angleArray=angleArray.reshape(angleArray.shape[0],1)

o_pickle = open("image.pk","wb")
pickle.dump(imageArray,o_pickle)
o_pickle.close()

o_pickle = open("angle.pk","wb")
pickle.dump(angleArray,o_pickle)
o_pickle.close()

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
encInput = tf.identity(conv5,"EncoderInput")
conv5 = tf.nn.relu(conv5)

conv5_reshaped = tf.reshape(conv5,shape = (-1,conv5.shape[1]*conv5.shape[2]*conv5.shape[3]))

dense1  = tf.compat.v1.layers.dense(conv5_reshaped,1048,activation=tf.nn.relu)
dense2  = tf.compat.v1.layers.dense(dense1,512,activation=tf.nn.relu)
dense3  = tf.compat.v1.layers.dense(dense2,256,activation=tf.nn.relu)
dense4  = tf.compat.v1.layers.dense(dense3,1,name="dense4LAyer")#,activation=tf.nn.relu)
preeictedAngle = tf.identity(dense4,"Angle")


residual_error = tf.pow(dense4-output,2)
cost = tf.reduce_mean(residual_error) 
loss_summary = tf.compat.v1.summary.scalar(name="loss",tensor=cost)
r_error = tf.compat.v1.summary.scalar(name="error",tensor=residual_error[0][0])

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(cost, name="optimizer")
#minimizer = tf.identity(train_op,"Minimkizer")

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "GURMUNI114487:7000")
    summ_writer = tf.compat.v1.summary.FileWriter(path+"out",sess.graph)
    init.run()
    stime = time.time()
    saver = tf.train.Saver()
    
    epoch = 10
    previous = 0
    for i in range(epoch):
        ls,_,loss = sess.run((loss_summary, train_op, cost), 
                                feed_dict={inputVal:imageArray, 
                                            output:angleArray,
                                            lr:0.001})
        print("***************************************************",loss,"i:", i) 
        summ_writer.add_summary(ls, i)
    saver.save(sess,path+"out/graph/tensorflowModel.ckpt")
    tf.train.write_graph(sess.graph.as_graph_def(), path+"out/graph", 'tensorflowModel.pbtxt', as_text=True)
    
    
    print(time.time()-stime)  
    freeze_graph.freeze_graph(path+'out/graph/tensorflowModel.pbtxt', "", 
                              False, 
                              path+'out/graph/tensorflowModel.ckpt', 
                              "Angle, EncoderInput, optimizer", #Since using optimizer, as well it will have
                              #the full graph details thats why size would be as same as "out/graph/tensorflowModel.ckpt'"
                              "save/restore_all", "save/Const:0", #not relevant
                              path+'out/graph/frozentensorflowModel.pb', 
                              True, ""  
                         )
    
    inputGraph = tf.GraphDef()
    with tf.gfile.Open(path+"/out/graph/frozentensorflowModel.pb","rb") as f:
        data2read = f.read()
        inputGraph.ParseFromString(data2read)
    outputGraph = optimize_for_inference_lib.optimize_for_inference(
                    inputGraph,
                    ["features"],
                    ["Angle", "EncoderInput"],#optimizer not considered, size is small.
                    tf.float32.as_datatype_enum)
    f = tf.gfile.FastGFile(path+'out/graph/OptimizedGraph.pb', "w")
    f.write(outputGraph.SerializeToString()) 
    