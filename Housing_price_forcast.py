from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

(train_data, test_data), (train_labels, test_labels) = boston_housing.load_data()

