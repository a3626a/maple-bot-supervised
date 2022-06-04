import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np

# [(x, y)]를 받고 동작-[(d up)*5] 을 반환한다.
def fit_model(features, labels, output_per_input):
    model = keras.Sequential([
        layers.LSTM(128),
        layers.Dense(lookup.vocab_size()*output_per_input)
    ])
    
    model.compile(loss = tf.losses.CategoricalCrossentropy(), optimizer = tf.optimizers.Adam(),
    validation_data=window.val)



    return model
