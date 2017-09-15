#!/usr/bin/env python

# imports
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


"""
Just some code to explore autoencoders. I will list the sources below.

Autoencoders areused to reconstruct the inputs. The two main applications are de-noising and dimensionality reduction.
It involves a two step process: 
1. Encoding: Mapping from input layer to the hidden layer. 
2. Decoding: Mapping from the hidden layer to the output layer.
The hidden layer can be thought of as a code that helps decode to obtain the original data. Hence, the
'code' must capture important features from the input data.

"""

# Load images
(X_train, _), (X_test, _) = mnist.load_data()

# Normalize and reshape
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


enc_len = 32
input_len = 784
input_image = Input(shape=(input_len,))
encoded = Dense(enc_len, activation = 'relu')(input_image)
decoded = Dense(input_len, activation = 'sigmoid')(encoded)
encoder = Model(input_image, encoded)
autoencoder = Model(input_image, decoded)
encoded_input = Input(shape=(enc_len,))
decoder_layer =  autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))


encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()