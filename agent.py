import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, img_width, model='models/15'):
        self.voc = ['', 'none', 'd down', 'down down', 'down up', 'f down', 'left down', 'left up', 'r down', 'right down', 'right up', 'up down', 'up up']
        self.model = tf.keras.models.load_model(model)
        self.img_width = img_width
        self.width = 40
        self.sampling_rate = 5
        # Check its architecture
        self.model.summary()

        self.imgs = []
        self.arrows = []

    def run(self, image, up, down, left, right):
        self.imgs.append(image)
        if len(self.imgs) > self.img_width :
            self.imgs.pop(0)

        self.arrows.append([up, down, left, right])
        if len(self.arrows) > self.img_width :
            self.arrows.pop(0)

        if len(self.imgs) < self.img_width :
            return None

        if len(self.arrows) < self.img_width :
            return None

        images = np.array([self.imgs], dtype=np.float16)
        arrows = np.array([self.arrows], dtype=np.float16)
        targets = np.array([[['grind']]*self.img_width])

        #return [self.voc[i] for i in list(np.argmax(self.model.predict(input).reshape(10,17), axis=1))]
        
        start = time.time()
        prd = self.model({'images': images, 'arrows': arrows, 'targets': targets})
        key = self.voc[np.argmax(prd, axis=1)[-1]]

        return key