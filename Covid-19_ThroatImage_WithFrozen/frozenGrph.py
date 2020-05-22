import time
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from enum import Enum

from tensorflow.python import debug as tf_debug
import time
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import tensorboard as tb
from tensorflow.python import ops
import pandas as pd
from skimage import io, transform
import numpy as np

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
test_X, test_Y = readData(testingData, "./ImageDataset/DataSetwithFiltereredout_Revisit/test/")
#train_X, train_Y = readData(trainingData, "./ImageDataset/DataSetwithFiltereredout_Revisit/train/")

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

def binaryConvertor(arr):
    binaryArray = []
    for (index,val) in enumerate(arr):
        if(arr[index]>=0.5):
            binaryArray.append(1)
        else:
            binaryArray.append(0)
    return np.asarray(binaryArray)

def frozenGraph():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format( "./trained_model_CNN/trained"))
        saver.restore(sess, tf.train.latest_checkpoint( "./trained_model_CNN/"))
        print("***********************saverafter restore:",saver)
        features = tf.compat.v1.get_default_graph().get_tensor_by_name('X:0')
        output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('Sigmoid:0')
        dropout_value = tf.compat.v1.get_default_graph().get_tensor_by_name('dropOut:0')
        
        for index,(data,labels) in enumerate(get_batchofImage(test_X,test_Y,200)):

            pred = sess.run((output_tensor), feed_dict={features: data, dropout_value:0.0})
            break
        #print(pred)
        pred = binaryConvertor(pred)
        print(confusion_matrix(labels,pred))
        print(recall_score(labels,pred))
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                            tf.compat.v1.get_default_graph().as_graph_def(),
                            ["Sigmoid","X","dropOut","totalLoss"]) 
 
        with tf.io.gfile.GFile("./trained_model_CNN/fgraph/frozentensorflowModel.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        
frozenGraph()    
    

def load_graph():
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open("./trained_model_CNN/fgraph/frozentensorflowModel.pb","rb") as f:
        graph_def.ParseFromString(f.read())
    
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def)
        
        #tf.import_graph_def(graph_def)
    
    return graph

def useFrozenGraph():
    graph = load_graph()
    input_name = "import/X"
    dropout = "import/dropOut"
    output = "import/Sigmoid"
    features = graph.get_operation_by_name(input_name);
    dropout_value = graph.get_operation_by_name(dropout)
    output_tensor = graph.get_operation_by_name(output)
    with tf.compat.v1.Session(graph=graph) as sess:
        for index,(data,labels) in enumerate(get_batchofImage(test_X,test_Y,200)):
            
            pred = sess.run(output_tensor.outputs[0], feed_dict={features.outputs[0]: data, 
                                                    dropout_value.outputs[0]:0.0})
            pred = binaryConvertor(pred)
            print(confusion_matrix(labels,pred))
            print(recall_score(labels,pred))
            
            break
     
        
            
useFrozenGraph()


def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format(
      [node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format(
      [node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format( 
      [node.name for node in graph_def.node if (
          'predictions' in node.name or 'softmax' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format(
      [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format(
      len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Identity'])))
  print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))
    
def get_graph_def_from_file(graph_filepath):
  with ops.Graph().as_default():
    with tf.io.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def    
    
describe_graph(get_graph_def_from_file("./trained_model_CNN/fgraph/frozentensorflowModel.pb"))    
    
def optimize_graph(model_dir, graph_filename, transforms, output_node):
  input_names = []
  output_names = [output_node]
  if graph_filename is None:
    graph_def = get_graph_def_from_saved_model(model_dir)
  else:
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name='optimized_model.pb')
  print('Graph optimized!')    
    
  
    
    
    