import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import horovod.tensorflow.keras as hvd
import numpy as np
import argparse
import time
import sys
from cifar import load_cifar
hvd.init()
batch_size = 16#args.batch_size
epochs = 100#args.epochs
#model_name = args.model_name
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(
    gpus[hvd.local_rank()], 'GPU')
    
train_ds, test_ds = load_cifar(batch_size)
model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=True, weights=None,
        input_shape=(128, 128, 3), classes=10)
if hvd.rank() == 0:
    print(model.summary())
opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
model.compile(
     loss='sparse_categorical_crossentropy',
     optimizer=opt,
     metrics=['accuracy'],
     experimental_run_tf_function=False)
callbacks = [
     hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]
if hvd.rank() == 0:
   verbose = 2
else:
   verbose=0
model.fit(train_ds, epochs=epochs, 
          verbose=verbose, callbacks=callbacks)