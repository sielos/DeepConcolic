import numpy as np
import tensorflow as tf
from tensorflow import keras

# model = keras.models.load_model('saved_models/cats_and_dogs.h5')
# model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'data/cat-dog/train'
test_path = 'data/cat-dog/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=300)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=300,
                         shuffle=False)

img_shape = 224, 224, 3

x_train, y_train = next(train_batches)
x_test, y_test = next(test_batches)

y_train = np.where(y_train == 1)[1]#.reshape(-1, 1)
y_test = np.where(y_test == 1)[1]#.reshape(-1, 1)

x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255


print(x_train)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


img_shape = 28, 28, 1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255

print(x_train)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
