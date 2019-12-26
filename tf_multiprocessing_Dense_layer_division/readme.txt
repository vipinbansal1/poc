This code will execute in two processes.
1. tf_multiprocessing_server.py is a server code which will create instances of two server(processes).
how to create two instances:
->python tf_multiprocessing_server.py 0  # 0 implies "localhost:2222"
->python tf_multiprocessing_server.py 1	 # 1 implies "localhost:2223"

How to run code in two different server or processes.
->python tf_encoder_multiprocessing.py

with tf.device("/job:local/task:1"):
execute code on server 1 i.e "localhost:2223"
Simillarly other one.

This approach can be used with GPU as well. 
In case of multiple GPU's and if some GPU have processing compabaility whereas another GPU has memory.
Then operations can be allocated as per the capability for faster execution.

tf_encoder_multiprocessing.py 
Divides the dense layer in to two parts....So far no performance improvement observed with this approach.