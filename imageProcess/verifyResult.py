#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : verify the accuracy of the model
 VERSION : 0.1
 DATE : 3.2017
"""

__author__ = 'Min'

import tensorflow as tf, sys

graphPath = sys.argv[1]
labelPath = sys.argv[2]
imagePath = sys.argv[3]

# read in the image
imageData = tf.gfile.FastGFile(imagePath, 'rb').read()

# loads label file
labelLines = [line.rstrip() for line in tf.gfile.GFile(labelPath)]

with tf.gfile.FastGFile(graphPath, 'rb') as f:
    graphDef = tf.GraphDef()
    graphDef.ParseFromString(f.read())
    _ = tf.import_graph_def(graphDef, name = '')

with tf.Session() as sess:
    softmaxTensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmaxTensor, \
                {'DecodeJpeg/contents:0': imageData})

    topK = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for nodeID in topK:
        className = labelLines[nodeID]
        ratio = predictions[0][nodeID]
        print('%s (ratio = %.5f)' % (className, ratio))
