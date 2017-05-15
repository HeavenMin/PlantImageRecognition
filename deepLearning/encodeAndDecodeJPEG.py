#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : encode and decode for JPEG image
 VERSION : 0.1
 DATE : 4.2017
"""

__author__ = 'Min'

import matplotlib.pyplot as plt
import tensorflow as tf

def decodeJPEG(rawData):
    return tf.image.decode_jpeg(rawData)

def encodeJPEG(imgData):
    return tf.image.encode_jpeg(imgData)


if __name__ == '__main__':
    IMAGE_PATH = input('enter your image path: ')[:-1]

    rawImage = tf.gfile.FastGFile(IMAGE_PATH, 'r').read()

    with tf.Session() as sess:

        imgData = encodeJPEG(rawImage)

        print(imgData.eval())

        plt.imshow(imgData.eval())
        plt.show()
        #
        # imgData = tf.image.convert_image_dtype(imgData, dtype = tf.float32)

        # encodedImage = tf.image.encode_jpeg(imgData)
