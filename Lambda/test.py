import tensorflow as tf
import os
import random
from skimage import io, transform
import numpy as np
from skimage.color import rgb2gray
import pandas as pd
import time
import horovod.tensorflow as hvd
hvd.init()
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.experimental.list_physical_devices('GPU')
print("***************************",gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("*******************",gpus[hvd.local_rank()])
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1, 'GPU':0})
config.gpu_options.allow_growth = True

config.gpu_options.visible_device_list = str("0,1")
print(config.gpu_options.visible_device_list )
print(os.environ["CUDA_VISIBLE_DEVICES"])