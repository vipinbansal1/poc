import tensorflow_datasets as tfds
import tensorflow as tf

import os
import horovod.tensorflow as hvd
print(tf.__version__)



hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']
#strategy = tf.distribute.MirroredStrategy()

#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)





#with strategy.scope():
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10)
])

opt = tf.keras.optimizers.Adam()
opt = hvd.DistributedOptimizer(opt)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer=opt,
			metrics=['accuracy'])
import time
start = time.time()
model.fit(train_dataset, epochs=12)
print("---------------------------------:",time.time()-start)