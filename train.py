import os
import time
import pickle
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers.experimental import preprocessing

from model import fit_model
from window_generator import WindowGenerator

matplotlib.use('tkagg')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['TF_GPU_THREAD_COUNT'] = '4'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=D:/Lib/cuda110'
os.environ['TF_XLA_FLAGS'] = ' --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

TRAIN_DIR = ['dataset/preprocessed/14/train/*']
VALID_DIR = ['dataset/preprocessed/14/valid/*']
BATCH_SIZE = 64
MAX_EPOCHS = 80
###
FILE_BATCH  = 1000
LABELS      = ['d down', 'down down', 'down up', 'f down', 'left down', 'left up', 'r down', 'right down', 'right up', 'up down', 'up up']
IMG_WIDTH   = 64 # 6.4 sec
DOWNSCALE   = 15
TARGETS     = ['grind', 'collect']
###

np.set_printoptions(precision=3, suppress=True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def visualize(train_ds):
    # test
    x, y = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for i in range(IMG_WIDTH):
        img = x['images'][0][i]
        ax = plt.subplot(8, 8, i + 1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def main():
    train_ds = WindowGenerator(
        batch_size=BATCH_SIZE,
        file_batch=FILE_BATCH,
        img_width=IMG_WIDTH,
        downscale=DOWNSCALE,
        labels=LABELS, 
        directories=TRAIN_DIR
    )
    train_ds = train_ds.make_dataset2()
    visualize(train_ds)

    valid_ds = WindowGenerator(
        batch_size=BATCH_SIZE,
        file_batch=FILE_BATCH,
        img_width=IMG_WIDTH,
        downscale=DOWNSCALE,
        labels=LABELS, 
        directories=VALID_DIR
    )
    valid_ds = valid_ds.make_dataset2()

    spec = train_ds.element_spec
    print(spec)

    image_inputs = keras.Input(shape=(IMG_WIDTH, 1080//DOWNSCALE, 1920//DOWNSCALE, 3), dtype=tf.float16)
    arrow_inputs = keras.Input(shape=(IMG_WIDTH, 4), dtype=tf.float16)
    target_inputs = keras.Input(shape=(IMG_WIDTH, ), dtype=tf.string)

    x1 = preprocessing.Rescaling(1./255)(image_inputs)
    # 72 x 128
    x1 = tf.reshape(x1, (-1, 1080//DOWNSCALE, 1920//DOWNSCALE, 3))
    x1 = layers.Conv2D(8, 4, strides=2, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Dropout(0.2)(x1)
    # 36 x 64 x 16
    x1 = layers.Conv2D(16, 4, strides=2, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Dropout(0.2)(x1)
    # 18 x 32 x 32
    x1 = layers.Conv2D(32, 4, strides=2, padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Dropout(0.2)(x1)

    x1 = tf.reshape(x1, (-1, IMG_WIDTH, 32*9*16))
    # 9 x 16 x 64

    x3 = layers.Dropout(0.2)(arrow_inputs)

    target_lookup = preprocessing.StringLookup(
        vocabulary=TARGETS
    )
    target_vocab_size = target_lookup.vocab_size()
    x5 = target_lookup(target_inputs)
    x5 = tf.reshape(x5, (-1, ))
    x5 = preprocessing.CategoryEncoding(max_tokens=target_vocab_size)(x5)
    x5 = tf.reshape(x5, (-1, IMG_WIDTH, int(target_vocab_size)))
    x5 = tf.cast(x5, dtype=tf.float16)

    x = tf.concat([x1, x3, x5], axis=2) # val acc 84.58
    x = layers.LSTM(1024, return_sequences=True)(x)
    x = tf.reshape(x, (-1, 1024))
    output = layers.Dense(len(LABELS)+2, activation=None)(x)

    model = keras.Model(inputs={'images':image_inputs, 'arrows':arrow_inputs, 'targets':target_inputs}, outputs=output)
    model.summary()

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True) # Enable XLA.
    model.compile(
        loss = tf.losses.CategoricalCrossentropy(
            from_logits=True
        ), 
        optimizer = tf.optimizers.Adam(learning_rate=0.000003),
        metrics=['accuracy']
    )

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = logs,
        histogram_freq = 1,
        # profile_batch = 0
        profile_batch=(50,60)
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/15',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    model.fit(
        train_ds, 
        epochs=MAX_EPOCHS,
        validation_data=valid_ds, 
        callbacks = [model_checkpoint_callback]
        # callbacks = [tboard_callback],
    )
    model.summary()

if __name__ == '__main__':
    main()