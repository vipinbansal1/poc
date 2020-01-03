# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:11:58 2020

@author: GUR45397
"""

import tensorflow as tf
import pandas as pd

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





def load_graph():
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(path+"/out/graph/OptimizedGraph.pb","rb") as f:
        graph_def.ParseFromString(f.read())
    
    with graph.as_default():
        tf.graph_util.import_graph_def(graph_def)
        
        #tf.import_graph_def(graph_def)
    
    return graph

graph = load_graph()

input_name = "import/features"
output_Angle_op = "import/Angle"
output_Encoder_features_op = "import/EncoderInput"

input_operation = graph.get_operation_by_name(input_name);
output_Angle = graph.get_operation_by_name(output_Angle_op);
output_Encoder_features = graph.get_operation_by_name(output_Encoder_features_op);

with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_Encoder_features.outputs[0],
                      {input_operation.outputs[0]: imageList})
    print(results)
    