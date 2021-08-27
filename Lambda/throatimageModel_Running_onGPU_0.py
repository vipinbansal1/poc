

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import random
from skimage import io, transform
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
import time
import horovod.tensorflow as hvd
trainingData = pd.read_csv("./ImageDataset/DataSetwithFiltereredout_Revisit/train.csv")

testingData = pd.read_csv("./ImageDataset/DataSetwithFiltereredout_Revisit/test.csv")

from PIL import Image
def readData(df, path):
    
    imageArray  = [] 
    labelArray = []
        
    for index, row in df.iterrows():
        imageFileName = row['Name']
        label = row['Label']
        labelArray.append(label)

        try:
            img = Image.open(path+imageFileName)
            #img = io.imread(path+imageFileName, plugin='matplotlib')
            #img = rgb2gray(img)
            #img = img.convert('RGB')
            img = transform.resize(np.array(img), (224, 224),anti_aliasing=True)#height,width
            #io.imshow(img) 
            #io.show()
            #return
            #img = img.reshape(img.shape[0],img.shape[1],1)
            imageArray.append(img)
            #Normalizing image data
            
        
        except Exception as e:
            print(e)
            pass
    return imageArray, labelArray

train_X, train_Y = readData(trainingData[:50], "./ImageDataset/DataSetwithFiltereredout_Revisit/train/")

test_X, test_Y = readData(testingData, "./ImageDataset/DataSetwithFiltereredout_Revisit/test/")

train_X[0].shape

learningRate = 0.001
dropOut = 0.1

height = 224
width = 224
num_classes = 1
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()

#with tf.device('/GPU:1'):
	

inputImage = tf.compat.v1.placeholder(tf.float32, shape=[None, height, width,3], name="X")
#label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="label")
label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="label")
l_rate = tf.compat.v1.placeholder(tf.float32, name = "learningRate")
d_out = tf.compat.v1.placeholder(tf.float32, name = "dropOut")


regularizer = tf.keras.regularizers.l2()

conv2d_1 = tf.compat.v1.layers.conv2d(inputImage, 
							   filters=4, 
							   kernel_size=[3,3],
							   strides=[1,1], 
							   padding='SAME', 
							   activation="tanh",
							   kernel_regularizer = regularizer,
							   name="conv2d_1")
conv2d_1_dropout = tf.nn.dropout(conv2d_1, rate=d_out)

conv2d_2 = tf.compat.v1.layers.conv2d(conv2d_1_dropout, 
							   filters=6, 
							   kernel_size=[3,3],
							   strides=[1,1], 
							   padding='SAME', 
							   activation="tanh",
							   kernel_regularizer = regularizer,
							   name="conv2d_2")
conv2d_2_dropout = tf.nn.dropout(conv2d_2, rate=d_out*2)

conv2d_3 = tf.compat.v1.layers.conv2d(conv2d_2_dropout, 
							   filters=8, 
							   kernel_size=[3,3],
							   strides=[1,1], 
							   padding='SAME', 
							   activation="tanh",
							   kernel_regularizer = regularizer,
							   name="conv2d_3")
conv2d_3_dropout = tf.nn.dropout(conv2d_3, rate=d_out*2)



flat_layer = tf.reshape(conv2d_3, shape=
						[-1, conv2d_3.shape[1]*conv2d_3.shape[2]*conv2d_3.shape[3]])
fc1 = tf.compat.v1.layers.dense(flat_layer, 32, activation=tf.nn.relu, 
								kernel_regularizer = regularizer,
								name="fc1")
fc1_dropout = tf.nn.dropout(fc1, rate=d_out*2)

fc2 = tf.compat.v1.layers.dense(fc1_dropout, 16, activation=tf.nn.relu, 
								kernel_regularizer = regularizer,
								name="fc2")
fc2_dropout = tf.nn.dropout(fc2, rate=d_out)


logit = tf.compat.v1.layers.dense(fc2_dropout,num_classes , 
								   kernel_regularizer = regularizer,
								   name="logit")


sigmoidal_ouput = tf.math.sigmoid(logit)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(label, tf.reshape(logit, shape=[-1]))

#Adding Regularization loss
vars   = tf.compat.v1.trainable_variables()
r_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars
					if 'bias' not in v.name ]) * 0.01

loss = tf.reduce_mean(xentropy)
losses = tf.math.add(loss,r_loss)

total_loss = tf.identity(losses, name="totalLoss")
optimizer = tf.compat.v1.train.AdamOptimizer(l_rate)




training_op = optimizer.minimize(total_loss)
global_step = tf.compat.v1.train.get_or_create_global_step()
init = tf.compat.v1.global_variables_initializer()





def binaryConvertor(arr):
	binaryArray = []
	for (index,val) in enumerate(arr):
		if(arr[index]>=0.5):
			binaryArray.append(1)
		else:
			binaryArray.append(0)
	return np.asarray(binaryArray)

print(logit,"\n","\n",total_loss, "\n", label, "\n", d_out, "\n",xentropy, "\n","\n","\n")

def get_batchofImage(X, y, bsize):
	for batch_i in range(0, 1 + (len(X) //bsize)):
		start_i = batch_i * bsize
		try:
			X_data = X[start_i:start_i + bsize]    
			y_data = y[start_i:start_i + bsize]      
		except IndexError:
			X_data = X[start_i:]
			y_data = y[start_i:]
		yield np.array(X_data), np.array(y_data)

n_epochs = 500
trainingLoss = []
testloss = []
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0, 'GPU':1})
config.gpu_options.allow_growth = True

config.gpu_options.visible_device_list = str("0")

#with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks,
#                                           config=config) as sess:
with tf.compat.v1.Session() as sess:
	init.run()
	saver = tf.compat.v1.train.Saver()
	start = time.time()
	for epoch in range(n_epochs):
		i = 0
		cost = 0 
		for index,(data,labels) in enumerate(get_batchofImage(train_X,train_Y,500)):
			#print(data.shape, labels.shape)
			if(len(data) == 0):
				continue
			l, prediction,_ = sess.run(( total_loss,sigmoidal_ouput, training_op), feed_dict={inputImage: data, label: labels, 
											 l_rate:learningRate,d_out:dropOut
											 })
			cost = cost+l
			
		cost = cost/index
		trainingLoss.append(np.round(cost,6))
		#print("training loss:........",np.round(cost,6))
		if(epoch%10==0):
		  #print("loss:  ", np.round(cost,6))
		  binary = binaryConvertor(prediction)
		  #print("\n\n\nTraining: labels:",np.column_stack((labels,  np.round(prediction,90), binary)))
		oldArray = []
		for index,(data,labels) in enumerate(get_batchofImage(test_X,test_Y,180)):

			l, out = sess.run((total_loss, sigmoidal_ouput), feed_dict={inputImage: data, label: labels, 
											  l_rate:learningRate, d_out:0.0
											  })
			pred = binaryConvertor(out)
			testloss.append(np.round(l,6))
			#oldArray = np.column_stack((oldArray, out))
			#print("test loss:*************************",np.round(l,6))
			#print("\n\n\Testing:",np.column_stack((labels, np.round(out,6), pred)))
			break
			# print(val.shape)
		   
		#acc_train = accuracy.eval(feed_dict={X: X_bach, y: y_batch})
		#acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
		print("Epoch***********************:",epoch)

		if(epoch%20==0):
		  print("\n\n\Testing:",np.column_stack((labels, np.round(out,6), pred)))
			
		  #print( "testloss:", testloss, sess, saver)
	print("Time ******************************",time.time()-start)
	try:
	  save_path = saver.save(sess, "./laptop/trained")
	  print("saved")
	except Exception as e:
	  print(e)


