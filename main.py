# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape, train_labels.shape)

#
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# model = keras.Sequential(
#     [keras.layers.Flatten(input_shape=(28, 28)),
#      keras.layers.Dense(128, activation='relu'),
#      keras.layers.Dense(10, activation='softmax')]
# )
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy']
#               )
#
# model.fit(train_images, train_labels)
#
# im = Image.open('test.jpeg')
# im1 = im.resize((28, 28))
# pix = im1.load()
# test_image = np.zeros(784).reshape(28, 28)
# for i in range(28):
#     for j in range(28):
#         r, g, b = pix[i, j]
#         bw = (r + g + b) // 3
#         pix[i, j] = bw, bw, bw
#         test_image[i][j] = (255 - bw)
#
# test_image = np.rot90(np.rot90(np.rot90(test_image / 255.0)))
# plt.imshow(test_image, cmap=plt.cm.binary)
# plt.show()
#
# test = (np.expand_dims(test_image, 0))
# print(test.shape)
# #
# # test_loss, test_acc = model.evaluate(test_images[0], test_labels[0:3])
# prediction = model.predict(test)
# print(np.argmax(prediction[0]), max(prediction[0]))
