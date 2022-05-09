import datasets
import numpy as np

def load_cifar10_data (**_):
  import tensorflow as tf
  img_shape = 32, 32, 3
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

datasets.register_dataset ('cifar10', load_cifar10_data)
