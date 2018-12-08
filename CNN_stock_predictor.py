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

TRAIN_SPLIT = .6
VALIDATION_SPLIT = .2
TEST_SPLIT = .2

def load_dataset(csv_path):

    print("*******************loading dataset*******************")

    data = pd.read_csv(csv_path, encoding = "ISO-8859-1");
    num_data_points = len(data.index);

    train_tweets = data['Tweet content'].values[0:int(num_data_points * TRAIN_SPLIT)]
    val_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    test_tweets = data['Tweet content'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]

    train_labels = data['increase'].values[0:int(num_data_points * TRAIN_SPLIT)]
    val_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT): int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT)]
    test_labels = data['increase'].values[int(num_data_points * TRAIN_SPLIT)+ int(num_data_points * VALIDATION_SPLIT):]

    return train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels

def main():
    print ("hello")
    train_tweets, val_tweets, test_tweets, train_labels, val_labels, test_labels = load_dataset("final_data/compiled_data.csv")

    batch_size = 128
    num_classes = 10
    epochs = 10

    img_x, img_y = 1, 1


    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    train_tweets = train_tweets.reshape(train_tweets.shape[0], img_x, img_y, 1)
    test_tweets = test_tweets.reshape(test_tweets.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    train_tweets = train_tweets.astype('float32')
    test_tweets = test_tweets.astype('float32')
    train_tweets /= 255
    test_tweets /= 255
    print('train_tweets shape:', train_tweets.shape)
    print(train_tweets.shape[0], 'train samples')
    print(test_tweets.shape[0], 'test samples')


    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    model.fit(train_tweets, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(test_tweets, test_labels),
              callbacks=[history])
    score = model.evaluate(test_tweets, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(range(1, 11), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == "__main__":
    main()
