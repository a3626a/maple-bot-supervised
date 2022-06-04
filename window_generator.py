import os
import pickle

from functools import reduce
from sys import getsizeof

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import Dataset

class WindowGenerator():
    def __init__(self, 
                 batch_size,
                 file_batch,
                 img_width,
                 downscale,
                 labels,
                 directories):
        self.batch_size = batch_size
        self.file_batch = file_batch
        self.img_width = img_width
        self.downscale = downscale
        self.labels = labels # none 이 포함되어 있지 않음
        self.num_labels = len(self.labels) + 2 # '', 'none' 이 있기 때문에 2 추가됨
        self.directories = directories
        # by repeating, we can reduce epoch preparation overhead.
        self.repeat = 1
        self.seed = 3333
        self.shuffle_size = 100
        self.min_samples = 4

    def read_and_parse(self, x):
        x = tf.io.read_file(x)
        feature_description = {
            'images': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'ups': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'downs': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'lefts': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'rights': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'targets': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'labels': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        x = tf.io.parse_single_example(x, feature_description)

        images = x['images']
        ups = x['ups']
        downs = x['downs']
        lefts = x['lefts']
        rights = x['rights']
        targets = x['targets']
        labels = x['labels']

        images = tf.io.parse_tensor(images, out_type=tf.float16)
        ups = tf.io.parse_tensor(ups, out_type=tf.float16)
        downs = tf.io.parse_tensor(downs, out_type=tf.float16)
        lefts = tf.io.parse_tensor(lefts, out_type=tf.float16)
        rights = tf.io.parse_tensor(rights, out_type=tf.float16)
        targets = tf.io.parse_tensor(targets, out_type=tf.string)
        labels = tf.io.parse_tensor(labels, out_type=tf.string)

        images = tf.reshape(images, (-1, 1080//self.downscale, 1920//self.downscale, 3))
        ups = tf.reshape(ups, (-1, 1))
        downs = tf.reshape(downs, (-1, 1))
        lefts = tf.reshape(lefts, (-1, 1))
        rights = tf.reshape(rights, (-1, 1))
        arrows = tf.concat([ups, downs, lefts, rights], axis=1)
        targets = tf.reshape(targets, (-1, ))
        labels = tf.reshape(labels, (-1, ))
        return {
            'images': images,
            'arrows': arrows,
            'targets': targets,
            'labels': labels
        }

    def class_func(self, x) :
        return x['labels'][-1]

    def batch_all(self, x) :
        return tf.data.Dataset.zip({
            'images': x['images'].batch(self.img_width),
            'arrows': x['arrows'].batch(self.img_width),
            'targets': x['targets'].batch(self.img_width),
            'labels': x['labels'].batch(self.img_width)
        })

    def make_dataset2(self):
        ds = Dataset.list_files(self.directories, shuffle=False, seed=self.seed)
        count = ds.cardinality()
        ds = ds.shuffle(count)

        lookup = preprocessing.StringLookup(
            vocabulary=self.labels,
            oov_token='none'
        )

        def window(filename):
            # 당장 기술적으로 images 이외의 피쳐를 이용하기 어렵기 때문에 일단 제거한다.
            dct = self.read_and_parse(filename)

            dct = {
                'images': dct['images'][:-1],
                'arrows': dct['arrows'][:-1],
                'targets': dct['targets'][:-1],
                'labels': lookup(dct['labels'][1:])
            }
            ds = Dataset.from_tensor_slices(dct)
            ds = ds.window(
                self.img_width, shift=1, stride=1, drop_remainder=True
            )
            ds = ds.flat_map(self.batch_all)
            return ds

        ds = ds.interleave(
            window,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size, drop_remainder=True)

        vocab_size = lookup.vocab_size()
        one_hot = preprocessing.CategoryEncoding(max_tokens=vocab_size)

        self.voc = lookup.get_vocabulary()
        print("VOCABULARY::")
        print(self.voc)

        def preprocess(x):
            y = x['labels']
            y = tf.reshape(y, (-1, ))
            y = one_hot(y)
            y = tf.reshape(y, (-1, int(vocab_size)))
            return ({'images': x['images'], 'arrows': x['arrows'], 'targets':x['targets']}, y)

        ds = ds.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # ds = ds.prefetch(tf.data.AUTOTUNE)
        # ds = ds.apply(tf.data.experimental.prefetch_to_device('gpu'))
        ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        ds = ds.prefetch(tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = 4
        ds = ds.with_options(options)

        return ds

    def make_dataset(self, valid=False):
        def read_and_parse(x):
            parts = tf.strings.split(x, os.path.sep)

            x = tf.io.read_file(x)
            feature_description = {
                'images': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'ups': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'downs': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'lefts': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'rights': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'targets': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'features2': tf.io.FixedLenFeature([], tf.string, default_value=''),
            }
            x = tf.io.parse_single_example(x, feature_description)
            images = x['images']
            ups = x['ups']
            downs = x['downs']
            lefts = x['lefts']
            rights = x['rights']
            targets = x['targets']
            features2 = x['features2']
            images = tf.io.parse_tensor(images, out_type=tf.float16)
            ups = tf.io.parse_tensor(ups, out_type=tf.float16)
            downs = tf.io.parse_tensor(downs, out_type=tf.float16)
            lefts = tf.io.parse_tensor(lefts, out_type=tf.float16)
            rights = tf.io.parse_tensor(rights, out_type=tf.float16)
            targets = tf.io.parse_tensor(targets, out_type=tf.string)
            features2 = tf.io.parse_tensor(features2, out_type=tf.string)
            images = tf.reshape(images, (-1, self.img_width, 1080//self.downscale, 1920//self.downscale, 3))
            ups = tf.reshape(ups, (-1, 1, 1))
            downs = tf.reshape(downs, (-1, 1, 1))
            lefts = tf.reshape(lefts, (-1, 1, 1))
            rights = tf.reshape(rights, (-1, 1, 1))
            arrows = tf.concat([ups, downs, lefts, rights], axis=2)
            targets = tf.reshape(targets, (-1, 1))
            features2 = tf.reshape(features2, (-1, self.key_width))
            return (images,arrows,features2,targets), tf.fill((tf.shape(images)[0],), parts[-2])

        def list_and_count(x):
            ds = Dataset.list_files(tf.strings.join([x, '/*']), shuffle=False, seed=self.seed)
            if valid :
                ds = ds.take(1)
            else:
                ds = ds.skip(1)
                ds = ds.repeat(self.repeat)
            if not valid :
                count = ds.cardinality()
                ds = ds.shuffle(count)

            ds = ds.map(
                read_and_parse,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            ds = ds.unbatch()
            ds = ds.take((self.min_samples - 2)*self.file_batch*self.repeat)
            return ds

        dataset = Dataset.list_files(self.directories, shuffle=False)
        dataset = tf.data.experimental.sample_from_datasets(
            list(dataset.map(list_and_count))
        )

        # Nevet do cache here
        # Caching greatly decreases dataset diversity. (because we only use portion of the dataset for major labels every epoch)
        # dataset = dataset.cache()
        if not valid :
            dataset = dataset.shuffle(self.shuffle_size)
        dataset = dataset.batch(self.batch_size)

        lookup = preprocessing.StringLookup(
            vocabulary=self.labels,
            oov_token='none'
        )
        vocab_size = lookup.vocab_size()
        one_hot = preprocessing.CategoryEncoding(max_tokens=vocab_size)

        self.voc = lookup.get_vocabulary()
        print("VOCABULARY::")
        print(self.voc)

        def preprocess(x, y):
            y = lookup(y)
            y = tf.reshape(y, (-1, ))
            y = one_hot(y)
            y = tf.reshape(y, (-1, int(vocab_size)))
            return (x, y)

        dataset = dataset.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        # dataset = dataset.apply(tf.data.experimental.prefetch_to_device('gpu'))
        options = tf.data.Options()
        options.experimental_threading.private_threadpool_size = 4
        dataset = dataset.with_options(options)

        return dataset