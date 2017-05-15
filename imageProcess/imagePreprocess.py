#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : resize the size of the image
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

def stretchImage(imgData, sizeX = 300, sizeY = 300, method = 0):
    # imgData : tf.float32 image Data
    # sizeX : int
    # sizeY : int
    # method : int : 0 Bilinear interoolation
    #                1 Nearest neighbor interpolation
    #                2 Bicubic interpolation
    #                3 Area interpolation
    return tf.image.resize_images(imgData, [sizeX, sizeY], method = method)

def cutOrPadImage(imgData, sizeX = 300, sizeY = 300):
    return tf.image.resize_image_with_crop_or_pad(imgData, sizeX, sizeY)

def centralCropImage(imgData, percentage = 0.8):
    # percentage : (0,1] real number
    return tf.image.central_crop(imgData, percentage)

def flipImage(imgData, method = 3):
    # method : Int : 0: flip updown
    #                1: filp left right
    #                2: transpose image
    if method == 0:
        return tf.image.flip_up_down(imgData)
    elif method == 1:
        return tf.image.flip_left_right(imgData)
    elif method == 2:
        return tf.image.transpose_image(imgData)
    else:
        return tf.image.random_flip_left_right(imgData)

def adjustBrightOfImage(imgData):
    # tf.image.adjust_brightness(imgData, 0.5)
    return tf.image.random_brightness(imgData, 0.5)

def adjustContrastOfImage(imgData):
    return tf.image.random_contrast(imgData, 0, 5)

def adjustHueOfImage(imgData):
    return tf.image.random_hue(imgData, 0.5)

def adjustSaturationOfImage(imgData):
    return tf.image.random_saturation(imgData, 0, 5)

def standardizationImage(imgData):
    return tf.image.per_image_standardization(imgData)

if __name__ == '__main__':
    IMAGE_PATH = input('enter your image path: ')[:-1]
    rawImage = tf.gfile.FastGFile(IMAGE_PATH, 'r').read()
    with tf.Session() as sess:
        imgData = decodeJPEG(rawImage)
        imgData = tf.image.convert_image_dtype(imgData, dtype = tf.float32)

        resizedImg = standardizationImage(imgData)

        print(resizedImg.get_shape())
        plt.imshow(resizedImg.eval())
        plt.show()
