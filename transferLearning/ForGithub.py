
import tensorflow as tf
import tensornets as nets
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#for loading and visualizing audio files
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from PIL import Image


eight = 40
width = 90
height = 3
from os import listdir

from os import listdir
files = listdir('D:/repo/transferLearningTF/tf_code/img')
from skimage import io, transform
from PIL import Image
def get_batch(arr, bsize):
    path = "./img/"
    for batch_i in range(0, 1 + (len(arr) //bsize)):
        start_i = batch_i * bsize
        try:
            batch = arr[start_i:start_i + bsize]      
        except IndexError:
            batch = arr[start_i:]
        image_list  = [] 
        label = []
        
        for i in range(len(batch)):
            imageFileName = batch[i]
            
            if(i%2 == 0):
                label.append(1)
            else:
                label.append(0)
            
            try:
                img = Image.open(path+imageFileName)
                img = img.convert('RGB')
                img = transform.resize(np.array(img), (224, 224),anti_aliasing=True)#height,width
                image_list.append(img)
            except Exception as e:
               print(e)
               pass
        yield np.array(image_list), np.array(label)

def binaryConvertor(arr):
    binaryArray = []
    for (index,val) in enumerate(arr):
        if(arr[index]>=0.5):
            binaryArray.append(1)
        else:
            binaryArray.append(0)
    return np.asarray(binaryArray)


tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
learningRate = 0.0005
dropOut = 0.10
activation_tensor = tf.nn.leaky_relu
act = "LeakyRelu"



x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
y = tf.compat.v1.placeholder(tf.float32, shape=(None), name='output_y')
l_rate = tf.compat.v1.placeholder(tf.float32, name="rate")

regularizer = tf.keras.regularizers.l2()
#will load VGG19 Architecture and set logit will return 20 classes
vgg19 = nets.VGG19(x, is_training=True, classes=20)
model = tf.identity(vgg19, name='logits')

'''
Utility method to print vgg19 architect.
vgg19.print_outputs()
vgg19.get_outputs()
vgg19.get_outputs(), came to know the tensor name of the last dropout layer which is
'vgg19/drop7/dropout/mul_1:0'
'''
lastFC = tf.compat.v1.get_default_graph().get_tensor_by_name('vgg19/drop7/dropout/mul_1:0')

'''
Taking the outcome of VGG19 model. It can take (vgg19/model) in case we want 
to use the logit outcome of vgg19.
Since I want to use last layer outcome(before logit) which is dropout layer, I will
pass lastFC
'''

fc1 = tf.compat.v1.layers.dense(lastFC, 512, 
                                   kernel_regularizer = regularizer,
                                   name="fc1")

logit = tf.compat.v1.layers.dense(fc1, 1, 
                                   kernel_regularizer = regularizer,
                                   name="fc2")
sigmoidal_ouput = tf.math.sigmoid(logit, 'output')
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = tf.reshape(logit, shape=[-1]))

vars   = tf.compat.v1.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * l_rate

loss = tf.reduce_mean(tf.math.square(xentropy))
totoal_loss = loss+lossL2

optimizer = tf.compat.v1.train.AdamOptimizer(l_rate)
training_op = optimizer.minimize(totoal_loss)


def training():
    epochs = 1
    save_model_path = './model/image_classification'
    print('Training...')
    with tf.compat.v1.Session() as sess:    
        # Initializing the whole architect variables including VGG19
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # used for loading Graph in tensorboard
        writer = tf.compat.v1.summary.FileWriter("./model/image_classification", sess.graph)
        writer.close()
        
        #Get the list of trainable variabls, including VGG19
        vars = tf.compat.v1.trainable_variables()
        
        #Will help in getting the value of trainable variable including vgg19
        vars_vals = sess.run(vars)
        
        '''
        Can see and print the actual weight values. Vgg19 pretrained weights not loaded,
        it will be some random initialization
        '''
        for var, val in zip(vars, vars_vals):
            if var.name == "vgg19/logits/weights:0":
                print(var.name) 
                break
        
        #Pretrained weights loaded
        sess.run(vgg19.pretrained())
        
        
        vars = tf.compat.v1.trainable_variables()
        vars_vals = sess.run(vars)
        '''
        VGG19 pretrained weights are loaded, now it will print the pretrained weightd value.
        which come ost same everytime.
        '''
        for var, val in zip(vars, vars_vals):
            #print("var: {}, value: {}".format(var.name, val)) 
            if var.name == "vgg19/logits/weights:0":
                print(var.name)        
                break
        
        # Training cycle
        print('starting training ... ')
        for epoch in range(epochs):
            # Loop over all batches
            for index,(data,labels) in enumerate(get_batch(files,5)):
                sess.run(training_op, {x: data, y: labels, l_rate:0.001})
                break
        
        #print the updated weights
        vars_vals = sess.run(vars)        
        for var, val in zip(vars, vars_vals):
            #print("var: {}, value: {}".format(var.name, val)) 
            if var.name == "vgg19/logits/weights:0":
                print(var.name) 
                break  
                
        # Save Model
        print("saving model")
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, save_model_path)

#training()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def loadAndprediction():
    tf.compat.v1.disable_eager_execution()

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        # Restore variables from disk.
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format("./model/image_classification"))
        saver.restore(sess, "./model/image_classification")
        features = tf.compat.v1.get_default_graph().get_tensor_by_name('input_x:0')
        sigmoidal_output = tf.compat.v1.get_default_graph().get_tensor_by_name('output:0')
        vars = tf.compat.v1.trainable_variables()
        print(vars) #some infos about variables...
        vars_vals = sess.run(vars)
        #Print the updated weights
        for var, val in zip(vars, vars_vals):
            if var.name == "vgg19/logits/weights:0":
                print(var.name) 
                break  

      
        print("Model restored.")
        for index,(data,labels) in enumerate(get_batch(files,90)):
            out = sess.run(sigmoidal_output, {features: data})
            
          
loadAndprediction()




