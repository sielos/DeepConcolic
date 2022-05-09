import datasets
import numpy as np

def load_imageNet_data (**_):
  import tensorflow as tf
  img_shape = 28, 28, 1
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ str (i) for i in range (0, 10) ]

datasets.register_dataset ('imageNet', load_imageNet_data)