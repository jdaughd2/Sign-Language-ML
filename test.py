import tensorflow as tf
import numpy as np
import os
import argparse

# Define command line arguments
# Run "python test.py -h" for help
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="specify test data", action="store", default="./datasets/sign_mnist_test.csv")
parser.add_argument("-m", "--model", help="specify model", action="store")
parser.add_argument("-p", "--prediction", help="output predictions", action="store_true")
args = parser.parse_args()

# Reads in test data, correctly shapes it, and shuffles it
def get_test(test_dir):
    import random

    print("Reading test data...")
    test = np.genfromtxt(test_dir,delimiter=',')[1:]

    X_test = [[i[1:]] for i in test]
    y_test = [i[:1] for i in test]
    test_data = list(zip(X_test, y_test))

    random.shuffle(test_data)
    X_test, y_test = zip(*test_data)
    X_test = np.asarray(X_test).reshape(-1, 28, 28, 1) / 255.0
    y_test = np.asarray(y_test).reshape(-1)

    return X_test, y_test

X_test, y_test = get_test(args.file)

# if model is undefined use latest saved model
if args.model is None:
    os.chdir("./models")
    args.model = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)[-1]

model = tf.keras.models.load_model(args.model)

# if prediction flag is set display prediction info
if args.prediction:
    predictions = model.predict([X_test])
    for i in range(len(X_test)):
        print(f"{i}\tValue: {chr(int(y_test[i])+97)}\tPrediction: {chr(np.argmax(predictions[i])+97)}")

val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"val_loss: {val_loss}\tval_accuracy: {val_accuracy}")
