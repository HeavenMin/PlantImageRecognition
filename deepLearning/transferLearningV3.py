#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : transferLearning
 VERSION : 0.2
 DATE : 4.2017
"""

__author__ = 'Min'

import argparse
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import tensorflow as tf
import numpy as np

# from datetime import datetime
from six.moves import urllib
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# image input tensor's name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# modify these information when you use this program
MODEL_DIR = ''
MODEL_FILE = ''
IMAGE_DIR = ''
OUTPUT_GRPAH = ''
OUTPUT_LABELS = ''
BOTTLENECK_DIR = ''
SUMMARIES_DIR = ''  # save logs

TRAINING_STEPS = 4000
LEARNING_RATE = 0.01
#______________________________________________________________________________

TESTING_PERCENT = 10
VALIDATION_PERCENT = 10
EVAL_STEP_INTERVAL = 10

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = -1
VALID_BATCH_SIZE = 100

FINAL_TENSOR_NAME = 'final_result'

# bottleneck tensor size 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# FLIP_LEFT_RIGHT = True
# RANDOM_BRIGHTNESS = 0.4

PRINT_MISCLASSIFIED_TEST_IMAGES = True

MODEL_INPUT_WIDTH = 299     # no use if not using distortions
MODEL_INPUT_HEIGHT = 299    # no use if not using distortions
MODEL_INPUT_DEPTH = 3       # no use if not using distortions

RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


def createImageLists(imageDir, testingPercentage, validationPercventage):
    if not gfile.Exists(imageDir):
        print("Image dir'" + imageDir +"'not found.'")
        return None
    result = {}
    subDirs = [x[0] for x in gfile.Walk(imageDir)]
    isRootDir = True
    for subDir in subDirs:
        if isRootDir:
            isRootDir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        fileList = []
        dirName = os.path.basename(subDir)
        if dirName == imageDir:
            continue
        print("Looking for images in '" + dirName + "'")
        for extension in extensions:
            fileGlob = os.path.join(imageDir, dirName, '*.' + extension)
            fileList.extend(gfile.Glob(fileGlob))
        if not fileList:
            print('No file found')
            continue
        labelName = re.sub(r'[^a-z0-9]+', ' ', dirName.lower())
        trainingImages = []
        testingImages =[]
        validationImages = []
        for fileName in fileList:
            baseName = os.path.basename(fileName)
            hashName = re.sub(r'_nohash_.*$', '', fileName)
            hashNameHased = hashlib.sha1(compat.as_bytes(hashName)).hexdigest()
            percentHash = ((int(hashNameHased, 16) %
                            (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                            (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentHash < validationPercventage:
                validationImages.append(baseName)
            elif percentHash < (testingPercentage + validationPercventage):
                testingImages.append(baseName)
            else:
                trainingImages.append(baseName)
        result[labelName] = {
            'dir': dirName,
            'training': trainingImages,
            'testing': testingImages,
            'validation': validationImages,
        }
    return result

def getImagePath(imageLists, labelName, index, imageDir, category):
    if labelName not in imageLists:
        tf.logging.fatal('Label does not exist %s.', labelName)
    label_lists = imageLists[labelName]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    cateoryList = label_lists[category]
    if not cateoryList:
        tf.logging.fatal('Label %s has no images in the category %s.',
                        labelName, category)
    modIndex = index % len(cateoryList)
    baseName = cateoryList[modIndex]
    subDir = label_lists['dir']
    fullPath = os.path.join(imageDir, subDir, baseName)
    return fullPath

def getBottleneckPath(imageLists, labelName, index, bottleneckDir, categoty):
    return getImagePath(imageLists, labelName, index, bottleneckDir,
                        categoty) + '.txt'

def createInceptionGraph():
    with tf.Graph().as_default() as graph:
        modelFileName = os.path.join(
                        MODEL_DIR, 'classify_image_graph_def.pb')
        with gfile.FastGFile(modelFileName, 'rb') as f:
            graphDef = tf.GraphDef()
            graphDef.ParseFromString(f.read())
            bottleneckTensor, jpegDataTensor, resizedInputTensor = (
                tf.import_graph_def(graphDef, name='', return_elements=[
                BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneckTensor, jpegDataTensor, resizedInputTensor

def runBottleneckOnImage(sess, imageData, ImageDataTensor, bottleneckTensor):
    bottleneckValues = sess.run(bottleneckTensor,
                        {ImageDataTensor: imageData})
    bottleneckValues = np.squeeze(bottleneckValues)
    return bottleneckValues

def isDirExist(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def writeListOfFloatsToFile(listOfFloats, filePath):
    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *listOfFloats)
    with open(filePath, 'wb') as f:
        f.write(s)

def readListOfFloatsFromFile(filePath):
    with open(filePath, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
        return list(s)

def createBottleneckFile(bottleneckPath, imageLists, labelName, index,
                         imageDir, category, sess, jpegDataTensor,
                         bottleneckTensor):
    print('Create bottleneck at ' + bottleneckPath)
    imagePath = getImagePath(imageLists, labelName, index,
                             imageDir, category)
    if not gfile.Exists(imagePath):
        tf.logging.fatal('File not exist %s', imagePath)
    imageData = gfile.FastGFile(imagePath, 'rb').read()
    try:
        bottleneckValues = runBottleneckOnImage(sess,
                           imageData, jpegDataTensor, bottleneckTensor)
    except:
        pass

    bottleneckString = ','.join(str(x) for x in bottleneckValues)
    with open(bottleneckPath, 'w') as f:
        f.write(bottleneckString)

def getOrCreateBottleneck(sess, imageLists, labelName, index,
                          imageDir, category, bottleneckDir,
                          jpegDataTensor, bottleneckTensor):
    labelLists = imageLists[labelName]
    subDir = labelLists['dir']
    subDirPath = os.path.join(bottleneckDir, subDir)
    isDirExist(subDirPath)
    bottleneckPath = getBottleneckPath(imageLists, labelName, index,
                                       bottleneckDir, category)
    if not os.path.exists(bottleneckPath):
        createBottleneckFile(bottleneckPath, imageLists, labelName, index,
                             imageDir, category, sess, jpegDataTensor,
                             bottleneckTensor)
    with open(bottleneckPath, 'r') as f:
        bottleneckString = f.read()
    error = False
    try:
        bottleneckValues = [float(x) for x in bottleneckString.split(',')]
    except ValueError:
        print('Invalid float, re-creating bottleneck')
        error = True
    if error:
        createBottleneckFile(bottleneckPath, imageLists, labelName, index,
                             imageDir, category, sess, jpegDataTensor,
                             bottleneckTensor)
        with open(bottleneckPath, 'r') as f:
            bottleneckString = f.read()
        bottleneckValues = [float(x) for x in bottleneckString.split(',')]

    return bottleneckValues

def cacheBottlenecks(sess, imageLists, imageDir, bottleneckDir,
                     jpegDataTensor, bottleneckTensor):
    bottlenecksNum = 0
    isDirExist(bottleneckDir)
    for labelName, labelLists in imageLists.items():
        for category in ['training', 'testing', 'validation']:
            categoryList = labelLists[category]
            for index, unusedBaseName in enumerate(categoryList):
                getOrCreateBottleneck(sess, imageLists, labelName, index,
                                      imageDir, category, bottleneckDir,
                                      jpegDataTensor, bottleneckTensor)
                bottlenecksNum += 1
                if bottlenecksNum % 100 == 0:
                    print(str(bottlenecksNum) + ' bottleneck created.')

def getRandomCachedBottlenecks(sess, imageLists, num, category,
                               bottleneckDir, imageDir,
                               jpegDataTensor, bottleneckTensor):
    classCount = len(imageLists.keys())
    bottlenecks = []
    groundTruths = []
    filenames = []
    if num >= 0:
        for nouse in range(num):
            labelIndex = random.randrange(classCount)
            labelName = list(imageLists.keys())[labelIndex]
            imageIndex = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            imageName = getImagePath(imageLists, labelName, imageIndex,
                                     imageDir, category)
            bottleneck = getOrCreateBottleneck(sess, imageLists, labelName,
                                               imageIndex, imageDir, category,
                                               bottleneckDir, jpegDataTensor,
                                               bottleneckTensor)
            groundTruth = np.zeros(classCount, dtype = np.float32)
            groundTruth[labelIndex] = 1.0
            bottlenecks.append(bottleneck)
            groundTruths.append(groundTruth)
            filenames.append(imageName)
    else:
        for labelIndex, labelName in enumerate(imageLists.keys()):
            for imageIndex, imageName in enumerate(imageLists[labelName][category]):
                imageName = getImagePath(imageLists, labelName, imageIndex,
                                         imageDir, category)
                bottleneck = getOrCreateBottleneck(sess, imageLists, labelName,
                                                   imageIndex, imageDir, category,
                                                   bottleneckDir, jpegDataTensor,
                                                   bottleneckTensor)
                groundTruth = np.zeros(classCount, dtype = np.float32)
                groundTruth[labelIndex] = 1.0
                bottlenecks.append(bottleneck)
                groundTruths.append(groundTruth)
                filenames.append(imageName)
    return bottlenecks, groundTruths, filenames

# for the crops, scales, flips after distortions
def getRandomDistortedBottlenecks(sess, imageLists, num, category, imageDir,
                                  inputJpegTensor, distortedImage,
                                  resizedInputTensor, bottleneckTensor):
    classCount = len(imageLists.keys())
    bottlenecks = []
    groundTruths = []
    for _ in range(num):
        labelIndex = random.randrange(classCount)
        labelName = list(imageLists.keys())[labelIndex]
        imageIndex = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        imagePath = getImagePath(imageLists, labelName, imageIndex,
                                 imageDir, category)
        if not gfile.Exists(imagePath):
            tf.logging.fatal('File not exist %s', imagePath)
        jpegData = gfile.FastGFile(imagePath, 'rb').read()
        distortedImageData = sess.run(distortedImage, {inputJpegTensor: jpegData})
        bottleneck = runBottleneckOnImage(sess, distortedImageData,
                                          resizedInputTensor, bottleneckTensor)
        groundTruth = np.zeros(classCount, dtype = np.float32)
        groundTruth[labelIndex] = 1.0
        bottlenecks.append(bottleneck)
        groundTruths.append(groundTruth)
    return bottlenecks, groundTruths

def tensorBoardUsage(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def finalTrainingLayer(classCount, finalTensorName, bottleneckTensor):
    with tf.name_scope('input'):
        bottleneckInput = tf.placeholder_with_default(
            bottleneckTensor, shape = [None, BOTTLENECK_TENSOR_SIZE],
            name = 'BottleneckInputPlaceholder')

    groundTruthInput = tf.placeholder(tf.float32,
                                      [None, classCount],
                                      name = 'GroundTruthInput')
    layerName = 'finalLayer'
    with tf.name_scope(layerName):
        with tf.name_scope('weights'):
            initialValue = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, classCount],
                                               stddev=0.001)
            layerWeights = tf.Variable(initialValue, name = 'finalWeights')
            tensorBoardUsage(layerWeights)
        with tf.name_scope('biases'):
            layerBiases = tf.Variable(tf.zeros([classCount]), name='finalBiases')
            tensorBoardUsage(layerBiases)
        with tf.name_scope('WxPlusB'):
            logits = tf.matmul(bottleneckInput, layerWeights) + layerBiases
            tf.summary.histogram('pre_activations', logits)

    finalTensor = tf.nn.softmax(logits, name=finalTensorName)
    tf.summary.histogram('activations', finalTensor)

    with tf.name_scope('crossEntropy'):
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(
                       labels=groundTruthInput, logits=logits)
        with tf.name_scope('total'):
            crossEntropyMean = tf.reduce_mean(crossEntropy)
    tf.summary.scalar('cross_entropy', crossEntropyMean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        trainStep = optimizer.minimize(crossEntropyMean)

    return (trainStep, crossEntropyMean, bottleneckInput, groundTruthInput,
            finalTensor)

def evaluation(resultTensor, groundTruthTensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correctPrediction'):
            prediction = tf.argmax(resultTensor, 1)
            correctPrediction = tf.equal(prediction,
                                         tf.argmax(groundTruthTensor, 1))
        with tf.name_scope('accuracy'):
            evaluation = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation)
    return evaluation, prediction

def main(_):

    graph, bottleneckTensor, jpegDataTensor, resizedImageTensor = createInceptionGraph()
    #load all images
    imageLists = createImageLists(IMAGE_DIR, TESTING_PERCENT, VALIDATION_PERCENT)
    #count the class number
    classCount = len(imageLists.keys())
    if classCount == 0 or classCount == 1:
        return

    with tf.Session(graph=graph) as sess:

        cacheBottlenecks(sess, imageLists, IMAGE_DIR,
                         BOTTLENECK_DIR, jpegDataTensor, bottleneckTensor)
        #add finaly layer
        (trainStep, crossEntropy, bottleneckInput, groundTruthInput,
         finalTensor) = finalTrainingLayer(classCount, FINAL_TENSOR_NAME,
                                           bottleneckTensor)
        evalutionStep, prediction = evaluation(finalTensor, groundTruthInput)

        merged = tf.summary.merge_all()
        trainWriter = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                            sess.graph)
        validationWriter = tf.summary.FileWriter(SUMMARIES_DIR + '/validation')

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(TRAINING_STEPS):
            trainBottlenecks, trainGroundTruth, _ = getRandomCachedBottlenecks(
                            sess,imageLists, TRAIN_BATCH_SIZE, 'training',
                            BOTTLENECK_DIR, IMAGE_DIR, jpegDataTensor,
                            bottleneckTensor)
            trainSummary, _ = sess.run([merged, trainStep],
                            feed_dict = {bottleneckInput: trainBottlenecks,
                                         groundTruthInput: trainGroundTruth})
            trainWriter.add_summary(trainSummary, i)

            isLastStep = ((i + 1) == TRAINING_STEPS)
            if (i % EVAL_STEP_INTERVAL) == 0 or isLastStep:
                trainAccuracy, crossEntropyValue = sess.run(
                            [evalutionStep, crossEntropy],
                            feed_dict = {bottleneckInput: trainBottlenecks,
                                         groundTruthInput: trainGroundTruth})
                print('Step %d: Train accuracy = %.1f%%' % (i, trainAccuracy * 100))
                print('Step %d: Cross entropy = %f' % (i, crossEntropyValue))

                validationBottlenecks, validationGroundTruth, _ = (
                            getRandomCachedBottlenecks(sess, imageLists,
                            VALID_BATCH_SIZE, 'validation', BOTTLENECK_DIR,
                            IMAGE_DIR, jpegDataTensor, bottleneckTensor))
                validationSummary, validationAccuracy = sess.run(
                            [merged, evalutionStep],
                            feed_dict = {bottleneckInput: validationBottlenecks,
                                        groundTruthInput: validationGroundTruth})
                validationWriter.add_summary(validationSummary, i)
                print('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                       (i, validationAccuracy * 100, len(validationBottlenecks)))

        print('@@__Here are final test evaluation__@@')

        testBottlenecks, testGroundTruth, testFileNames = (
                getRandomCachedBottlenecks(sess, imageLists, TEST_BATCH_SIZE,
                'testing', BOTTLENECK_DIR, IMAGE_DIR, jpegDataTensor,
                bottleneckTensor))
        testAccuracy, predictions = sess.run([evalutionStep, prediction],
                            feed_dict = {bottleneckInput: testBottlenecks,
                                         groundTruthInput: testGroundTruth})
        print('Final test accuracy = %.1f%% (N=%d)' % (testAccuracy * 100,
                                                      len(testBottlenecks)))

        if PRINT_MISCLASSIFIED_TEST_IMAGES:
            print('@@__Misclassified test images__@@')
            for i, testFileName in enumerate(testFileNames):
                if predictions[i] != testGroundTruth[i].argmax():
                    print('%70s  %s' % (testFileName,
                                        list(imageLists.keys())[predictions[i]]))

        outputGraphDef = graph_util.convert_variables_to_constants(sess,
                         graph.as_graph_def(), [FINAL_TENSOR_NAME])
        with gfile.FastGFile(OUTPUT_GRPAH, 'wb') as f:
            f.write(outputGraphDef.SerializeToString())
        with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
            f.write('\n'.join(imageLists.keys()) + '\n')

if __name__ == '__main__':
    tf.app.run()
