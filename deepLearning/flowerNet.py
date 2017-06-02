#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 AUTHOR : MIN
 PURPOSE : the deep learning CNN model, similar as inception
 VERSION : 0.1
 DATE : 4.2017
"""

__author__ = 'Min'


import math
import time
import tensorflow as tf
from datetime import datetime

NUM_CLASSES = 50

slim = tf.contrib.slim
# 产生截断的正太分布
# produce a truncated normal distriubtion
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 生成默认参数
# Generate default parameters
def flowerNet_arg_scope(weight_decay = 0.00004, stddev = 0.1,
                            batch_norm_var_collection = 'moving_vars'):
    batch_norm_params = {
        # 参数衰减系数
        # parameter attenuation coefficient
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # auto assign default values
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
            weights_regularizer = slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
            activation_fn = tf.nn.relu,     #Activation function
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as scope:
                return scope



# 生成网络的卷积 池化部分
# generate convolitonal layer and pooling layer in the CNN
def flowerNet_base(inputs, scope = None):
    end_points = {}
    with tf.variable_scope(scope, 'Inception', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride = 1, padding = 'VALID'):
            # convolutional layer 3x3/2 32
            net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'conv')
            # convolutional layer 3x3/1 32
            net = slim.conv2d(net, 32, [3, 3], scope = 'conv_1')
            # convolutional layer 3x3/1 64
            net = slim.conv2d(net, 64, [3, 3], padding= 'SAME',
                                scope = 'conv_2')
            # max pool layer 3x3/2
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool')
            # convolutional layer 1x1/1 80
            net = slim.conv2d(net, 80, [1, 1], scope = 'conv_3')
            # convolutional layer 3x3/1 192
            net = slim.conv2d(net, 192, [3, 3], scope = 'conv_4')
            # max pool layer 3,3/2
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool_1')

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride = 1, padding = 'SAME'):
            # mixed module 1
            with tf.variable_scope('mixed'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 48, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 64, [5, 5], scope = 'conv1')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv2')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 32, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 2
            with tf.variable_scope('mixed_1'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 48, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 64, [5, 5], scope = 'conv1')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv2')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 64, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 3
            with tf.variable_scope('mixed_2'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 48, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 64, [5, 5], scope = 'conv1')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 96, [3, 3], scope = 'conv2')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 64, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 4
            with tf.variable_scope('mixed_3'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 384, [3, 3], stride = 2,
                                        padding = 'VALID', scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 64, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 96, [3, 3], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 96, [3, 3], stride = 2,
                                          padding = 'VALID', scope = 'conv2')
                with tf.variable_scope('branch2'):
                    branch2 = slim.max_pool2d(net, [3, 3], stride = 2,
                                             padding = 'VALID', scope = 'maxPool')
                net = tf.concat([branch0, branch1, branch2], 3)

            # mixed module 5
            with tf.variable_scope('mixed_4'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 128, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 128, [1, 7], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 192, [7, 1], scope = 'conv2')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 128, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 128, [7, 1], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 128, [1, 7], scope = 'conv2')
                    branch2 = slim.conv2d(branch2, 128, [7, 1], scope = 'conv3')
                    branch2 = slim.conv2d(branch2, 192, [1, 7], scope = 'conv4')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 6
            with tf.variable_scope('mixed_5'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 160, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 160, [1, 7], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 192, [7, 1], scope = 'conv2')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 160, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 160, [7, 1], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 160, [1, 7], scope = 'conv2')
                    branch2 = slim.conv2d(branch2, 160, [7, 1], scope = 'conv3')
                    branch2 = slim.conv2d(branch2, 192, [1, 7], scope = 'conv4')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 7
            with tf.variable_scope('mixed_6'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 160, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 160, [1, 7], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 192, [7, 1], scope = 'conv2')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 160, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 160, [7, 1], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 160, [1, 7], scope = 'conv2')
                    branch2 = slim.conv2d(branch2, 160, [7, 1], scope = 'conv3')
                    branch2 = slim.conv2d(branch2, 192, [1, 7], scope = 'conv4')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 8
            with tf.variable_scope('mixed_7'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 192, [1, 7], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 192, [7, 1], scope = 'conv2')
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 192, [7, 1], scope = 'conv1')
                    branch2 = slim.conv2d(branch2, 192, [1, 7], scope = 'conv2')
                    branch2 = slim.conv2d(branch2, 192, [7, 1], scope = 'conv3')
                    branch2 = slim.conv2d(branch2, 192, [1, 7], scope = 'conv4')
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)
            end_points['mixed_7'] = net

            # mixed module 9
            with tf.variable_scope('mixed_8'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                    branch0 = slim.conv2d(branch0, 320, [3, 3], stride = 2,
                                          padding = 'VALID', scope = 'conv1')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 192, [1, 1], scope = 'conv')
                    branch1 = slim.conv2d(branch1, 192, [1, 7], scope = 'conv1')
                    branch1 = slim.conv2d(branch1, 192, [7, 1], scope = 'conv2')
                    branch1 = slim.conv2d(branch1, 192, [3, 3], stride = 2,
                                          padding = 'VALID', scope = 'conv3')
                with tf.variable_scope('branch2'):
                    branch2 = slim.max_pool2d(net, [3, 3], stride = 2,
                                             padding = 'VALID', scope = 'maxPool')

                net = tf.concat([branch0, branch1, branch2], 3)

            # mixed module 10
            with tf.variable_scope('mixed_9'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 320, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 384, [1, 1], scope = 'conv')
                    branch1 = tf.concat([
                        slim.conv2d(branch1, 384, [1, 3], scope = 'conv1'),
                        slim.conv2d(branch1, 384, [3, 1], scope = 'conv2')], 3)
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 448, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 384, [3, 3], scope = 'conv1')
                    branch2 = tf.concat([
                        slim.conv2d(branch2, 384, [1, 3], scope = 'conv2'),
                        slim.conv2d(branch2, 384, [3, 1], scope = 'conv3')], 3)
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            # mixed module 11
            with tf.variable_scope('mixed_10'):
                with tf.variable_scope('branch0'):
                    branch0 = slim.conv2d(net, 320, [1, 1], scope = 'conv')
                with tf.variable_scope('branch1'):
                    branch1 = slim.conv2d(net, 384, [1, 1], scope = 'conv')
                    branch1 = tf.concat([
                        slim.conv2d(branch1, 384, [1, 3], scope = 'conv1'),
                        slim.conv2d(branch1, 384, [3, 1], scope = 'conv2')], 3)
                with tf.variable_scope('branch2'):
                    branch2 = slim.conv2d(net, 448, [1, 1], scope = 'conv')
                    branch2 = slim.conv2d(branch2, 384, [3, 3], scope = 'conv1')
                    branch2 = tf.concat([
                        slim.conv2d(branch2, 384, [1, 3], scope = 'conv2'),
                        slim.conv2d(branch2, 384, [3, 1], scope = 'conv3')], 3)
                with tf.variable_scope('branch3'):
                    branch3 = slim.avg_pool2d(net, [3, 3], scope = 'avgPool')
                    branch3 = slim.conv2d(branch3, 192, [1, 1], scope = 'conv')

                net = tf.concat([branch0, branch1, branch2, branch3], 3)

            return net, end_points

# global avg pool and softmax and logits
def flowerNet(inputs, numClasses, isTraining = True,
              dropoutKeepProb = 0.8, predictionFN = slim.softmax,
              spatialSqueeze = True, reuse = None, scope = 'flowerNet'):
    with tf.variable_scope(scope, 'flowerNet', [inputs, numClasses],
                           reuse = reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training = isTraining):
            net, end_points = flowerNet_base(inputs, scope = scope)

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride = 1, padding = 'SAME'):
                aux_logits = end_points['mixed_7']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits,
                                                 [5, 5], stride = 3,
                                                 padding = 'VALID', scope = 'avgPool')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'conv')
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5],
                                        weights_initializer = trunc_normal(0.01),
                                        padding = 'VALID', scope = 'conv1')
                    aux_logits = slim.conv2d(aux_logits, numClasses, [1, 1],
                                             activation_fn = None,
                                             normalizer_fn = None,
                                             weights_initializer = trunc_normal(0.001),
                                             scope = 'conv2')
                    if spatialSqueeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')

                    end_points['AuxLogits'] = aux_logits

                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8], padding = 'VALID',
                                          scope = 'avgPool')
                    net = slim.dropout(net, keep_prob = dropoutKeepProb,
                                       scope = 'dropout')
                    end_points['PreLogits'] = net

                    logits = slim.conv2d(net, numClasses, [1, 1], activation_fn = None,
                                         normalizer_fn = None, scope = 'conv')
                    if spatialSqueeze:
                        logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')

                end_points['Logits'] = logits
                end_points['Predictions'] = predictionFN(logits, scope = 'Predictions')

    return logits, end_points


def time_test(session, target, info_string):
    num_steps_burn = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn):
        startTime = time.time()
        _ = session.run(target)
        duration = time.time() - startTime
        if i >= num_steps_burn:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(),
                      i - num_steps_burn, duration))
                total_duration += duration
                total_duration_squared += duration * duration
                mn = total_duration / num_batches
                vr =total_duration_squared / num_batches - mn * mn
                sd = math.sqrt(vr)
                print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
                      (datetime.now(), info_string, num_batches, mn, sd))

if __name__ == '__main__':
    batchSize = 100
    height, weight = 299, 299
    inputs = tf.random_uniform((batchSize, height, weight, 3))

    with slim.arg_scope(flowerNet_arg_scope()):
        logits, end_points = flowerNet(inputs, isTraining = False,
                                       numClasses = NUM_CLASSES)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        num_batches = 100
        time_test(sess, logits, "Forward")









#END
