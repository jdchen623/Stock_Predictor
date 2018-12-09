from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import collections
import numpy as np
import util
import pandas as pd
from data_helpers import load_data


TRAIN_SPLIT = .6
VALIDATION_SPLIT = .2
TEST_SPLIT = .2



def main():
    print('Loading data')
    x, y, vocabulary, vocabulary_inv = load_data()

    # x.shape -> (10662, 56)
    # y.shape -> (10662, 2)
    # len(vocabulary) -> 18765
    # len(vocabulary_inv) -> 18765

    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

    # X_train.shape -> (8529, 56)
    # y_train.shape -> (8529, 2)
    # X_test.shape -> (2133, 56)
    # y_test.shape -> (2133, 2)


    sequence_length = x.shape[1] # 56
    vocabulary_size = len(vocabulary_inv) # 18765
    embedding_dim = 256
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5

    epochs = 100
    batch_size = 30

    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training







# train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels = load_dataset("final_data/compiled_data.csv")
#
# batch_size = 128
# num_classes = 10
# epochs = 10
#
# img_x, img_y = 1, 1
#
#
# # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
# train_tweets = train_tweets.reshape(train_tweets.shape[0], img_x, img_y, 1)
# test_tweets = test_tweets.reshape(test_tweets.shape[0], img_x, img_y, 1)
# input_shape = (img_x, img_y, 1)
#
# # convert the data to the right type
# train_tweets = train_tweets.astype('float32')
# test_tweets = test_tweets.astype('float32')
# train_tweets /= 255
# test_tweets /= 255
# print('train_tweets shape:', train_tweets.shape)
# print(train_tweets.shape[0], 'train samples')
# print(test_tweets.shape[0], 'test samples')
#
#
# # convert class vectors to binary class matrices - this is for use in the
# # categorical_crossentropy loss below
# train_labels = keras.utils.to_categorical(train_labels, num_classes)
# test_labels = keras.utils.to_categorical(test_labels, num_classes)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
#
#
# class AccuracyHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.acc = []
#
#     def on_epoch_end(self, batch, logs={}):
#         self.acc.append(logs.get('acc'))
#
# history = AccuracyHistory()
#
# model.fit(train_tweets, train_labels,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(test_tweets, test_labels),
#           callbacks=[history])
# score = model.evaluate(test_tweets, test_labels, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
if __name__ == "__main__":
    main()
