#This is a multiclass classification.

from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# print(len(train_data))
# print(len(test_data))
# print(train_data[2843])

#decoding function
# def decoding(x):
#     word_index = reuters.get_word_index()
#     reverse_word_index = dict(
#         [(value, key) for (key, value) in word_index.items()]
#     )
#     decoded_newswire = " ".join(
#         [reverse_word_index.get(i - 3, "?") for i in train_data[x]]
#     )
#     print(decoded_newswire)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#one hot coding:
#each label is a zero vector, only the element corresponding to the index of this tag is 1.
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

# y_train = to_one_hot(train_labels)
# y_test = to_one_hot(test_labels)

#or use a built-in method for keras
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:1000]
y_val = y_train[:1000]
partial_x_train = x_train[1000:]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(loss) + 1)
# plt.subplot(3, 1, 1)
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
#
# plt.subplot(3, 1, 3)
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
# plt.plot(epochs, acc, "bo", label="Training accuracy")
# plt.plot(epochs, val_acc, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
#
# plt.show()

# results = model.evaluate(x_test, y_test)
# print(results)

def pridict_result(x, x_test):
    predictions = model.predict(x_test)
    return np.argmax(predictions[x])

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)

# def Hand_test():
#     print("Please input the index you want to test:")
#     x = int(input())
#     print("The prediction is: ", pridict_result(x, x_test))
#     print("the correct answer is: ", np.argwhere(x_test == 1))
#
# Hand_test()

