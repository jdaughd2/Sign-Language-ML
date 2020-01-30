import tensorflow as tf
import numpy as np
import os

# Reads in test data, correctly shapes it, and shuffles it
def get_test():
    import random

    print("Reading test data...")
    test = np.genfromtxt('./datasets/sign_mnist_test/sign_mnist_test.csv',delimiter=',')[1:]

    X_test = [[i[1:]] for i in test]
    y_test = [i[:1] for i in test]
    test_data = list(zip(X_test, y_test))

    random.shuffle(test_data)
    X_test, y_test = zip(*test_data)
    X_test = np.asarray(X_test).reshape(-1, 28, 28, 1) / 255.0
    y_test = np.asarray(y_test).reshape(-1)

    return X_test, y_test

X_test, y_test = get_test()

# Use most recent model
os.chdir("./models")
models = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
model = tf.keras.models.load_model(models[-1])
# Uncomment to use specific model
#model = tf.keras.models.load_model("./models/{model}")


val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"val_loss: {val_loss}\tval_accuracy: {val_accuracy}")
