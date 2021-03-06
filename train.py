import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import os
import argparse

# Define command line arguments
# Run "python test.py -h" for help
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="specify test data", action="store", default="./datasets/sign_mnist_train.csv")
parser.add_argument("-e", "--epoch", help="specify number of epochs", type=int, default=10)
args = parser.parse_args()

# Reads in training data, correctly shapes it, and shuffles it
def get_train(train_dir):
    import random

    print("Reading training data...")
    train = np.genfromtxt(train_dir,delimiter=',')[1:]

    X_train = [[i[1:]] for i in train]
    y_train = [i[:1] for i in train]
    training_data = list(zip(X_train, y_train))

    random.shuffle(training_data)
    X_train, y_train = zip(*training_data)
    X_train = np.asarray(X_train).reshape(-1, 28, 28, 1) / 255.0
    y_train = np.asarray(y_train).reshape(-1)

    return X_train, y_train

if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("datasets/images"):
    os.makedirs("datasets/images")

NAME = f"sign-language-cnn-64x2-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"./logs/{NAME}")

X_train, y_train = get_train(args.file)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(25))
model.add(Activation("sigmoid"))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=args.epoch, callbacks=[tensorboard])

model.save(f"./models/{NAME}.h5")
