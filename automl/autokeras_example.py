import numpy as np
import tensorflow as tf
from keras.datasets import mnist

import autokeras as ak
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]
print(x_train.shape)  # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_train[:3])  # array([7, 2, 1], dtype=uint8)



# Initialize the image classifier.
clf = ak.ImageClassifier(overwrite=True, max_trials=5)
# Feed the image classifier with training data.
clf.fit(x_train, y_train)


# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))