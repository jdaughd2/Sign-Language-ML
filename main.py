import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np
import os

# Reads in training data, correctly shapes it, and shuffles it
def get_train():
    import random

    train = np.genfromtxt('./datasets/sign_mnist_train/sign_mnist_train.csv',delimiter=',')[1:]

    X_train = [[i[1:]] for i in train]
    y_train = [i[:1] for i in train]
    training_data = list(zip(X_train, y_train))

    random.shuffle(training_data)
    X_train, y_train = zip(*training_data)
    X_train = np.asarray(X_train).reshape(-1, 28, 28, 1) / 255.0
    y_train = np.asarray(y_train).reshape(-1)

    return X_train, y_train

# Reads in test data, correctly shapes it, and shuffles it
def get_test():
    import random

    test = np.genfromtxt('./datasets/sign_mnist_test/sign_mnist_test.csv',delimiter=',')[1:]

    X_test = [[i[1:]] for i in test]
    y_test = [i[:1] for i in test]
    test_data = list(zip(X_test, y_test))

    random.shuffle(test_data)
    X_test, y_test = zip(*test_data)
    X_test = np.asarray(X_test).reshape(-1, 28, 28, 1) / 255.0
    y_test = np.asarray(y_test).reshape(-1)

    return X_test, y_test

if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("models"):
    os.makedirs("models")

NAME = f"sign-language-cnn-64x2-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"./logs/{NAME}")

X_train, y_train = get_train()

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

model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard])

X_test, y_test = get_test()
val_loss, val_accuracy = model.evaluate(X_test, y_test, callbacks=[tensorboard])
print(f"val_loss: {val_loss}\tval_accuracy: {val_accuracy}")

model.save("./models/signlanguage-cnn-64x2-{}.h5".format(int(time.time())))
