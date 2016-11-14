import os,sys
sys.path.append('./utils')

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from augmentation import geoAugmentation
from augmentation import photoAugmentation
from warpflow import loss_interp

def augmentation(source, target, mean):
    geo_source, geo_target = geoAugmentation(source, target)
    photo_source, photo_target = photoAugmentation(geo_source, geo_target, mean)
    return geo_source, geo_target, photo_source, photo_target

def keepOrigin(source, target):
    # Return as they are for test
    return source, target, source, target

# TODO: inception model is not ready yet
def inception_v3_base(inputs,
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            #print end_point, net.get_shape()
            end_points[end_point] = net
            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            #print end_point, net.get_shape()
            end_points[end_point] = net
            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            #print end_point, net.get_shape()
            end_points[end_point] = net
            # 35 x 35 x 192.

        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                         scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                         scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                         scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                         scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                         scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                         scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                         scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                         scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                         scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                     scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                     scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                     scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                     scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            end_points[end_point] = net
            

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                     padding='SAME', scope='Conv2d_1a_1x1')    # VALID
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                     scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                     padding='SAME', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME',
                                         scope='MaxPool_1a_3x3')
                net = tf.concat(3, [branch_0, branch_1, branch_2])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
    

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                     scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                     scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                     scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                     scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                     scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                     scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                     scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                     scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                     scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                     scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                     scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                     scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                     scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                     scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                     scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                     scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                     scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                     scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                     scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                     scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                     scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                     scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                     scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                     scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                     scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                     scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                     scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                     scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            end_points[end_point] = net

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'          # VALID
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                     padding='SAME', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                     scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                     scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                     padding='SAME', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME',
                                         scope='MaxPool_1a_3x3')
                net = tf.concat(3, [branch_0, branch_1, branch_2])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(3, [
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(3, [
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            # end_points[end_point] = net
            

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(3, [
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(3, [
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
            #print end_point, net.get_shape()
            end_points[end_point] = net
    return net, end_points
    

def inception_v3(inputs, 
                 outputs, 
                 loss_weight,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):

    mean = tf.constant([97.533268117955444, 99.238235788550085, 97.055973199626948], dtype=tf.float32, name="img_global_mean")
    # tf.tile(mean, [4,192,256,1])
    inputs = inputs - mean
    outputs = outputs - mean
    # Scaling to 0 ~ 1 or -0.4 ~ 0.6?
    inputs = tf.truediv(inputs, 255.0)
    outputs = tf.truediv(outputs, 255.0)
    network_in = tf.concat(3, [inputs, outputs])

    # Add local response normalization (ACROSS_CHANNELS) for computing photometric loss
    inputs_norm = tf.nn.local_response_normalization(inputs, depth_radius=4, beta=0.7)
    outputs_norm = tf.nn.local_response_normalization(outputs, depth_radius=4, beta=0.7)

    # end_points will collect relevant activations for external use, for example summaries or losses.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [network_in, num_classes],
                         reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
            _, end_points = inception_v3_base(network_in, scope=scope, min_depth=min_depth,depth_multiplier=depth_multiplier)
            
            # Hyper-params for computing unsupervised loss
            epsilon = 0.0001 
            alpha_c = 0.25
            alpha_s = 0.37
            lambda_smooth = 1.0
            deltaWeights = {}
            deltaWeights["FlowDeltaWeights"] = tf.constant([0,0,0,0,1,-1,0,0,0,0,0,0,0,1,0,0,-1,0], dtype=tf.float32, shape=[3,3,2,2], name="FlowDeltaWeights")
            #deltaWeights["ImgDeltaWeights"] = tf.constant([-1,0,1,-2,0,2,-1,0,1,-1,-2,-1,0,0,0,1,2,1], dtype=tf.float32, shape=[3,3,1,2], name="ImgDeltaWeights")
            #deltaWeights["mean"] = mean
            scale = 2       # for deconvolution

            # Expanding part
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                        activation_fn=tf.nn.elu): 
                pr6 = slim.conv2d(end_points["Mixed_7c"], 2, [3, 3], activation_fn=None, scope='pr6')
                h6 = pr6.get_shape()[1].value
                w6 = pr6.get_shape()[2].value
                pr6_input = tf.image.resize_bilinear(inputs_norm, [h6, w6])
                pr6_output = tf.image.resize_bilinear(outputs_norm, [h6, w6])
                flow_scale_6 = 0.625    # (*20/64)
                loss6, _ = loss_interp(pr6, pr6_input, pr6_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_6, deltaWeights)
                upconv5 = slim.conv2d_transpose(end_points["Mixed_7c"], 512, [2*scale, 2*scale], stride=scale, scope='upconv5')
                pr6to5 = slim.conv2d_transpose(pr6, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr6to5')
                concat5 = tf.concat(3, [end_points["Mixed_6e"], upconv5, pr6to5])

                pr5 = slim.conv2d(concat5, 2, [3, 3], activation_fn=None, scope='pr5')
                h5 = pr5.get_shape()[1].value
                w5 = pr5.get_shape()[2].value
                pr5_input = tf.image.resize_bilinear(inputs_norm, [h5, w5])
                pr5_output = tf.image.resize_bilinear(outputs_norm, [h5, w5])
                flow_scale_5 = 1.25    # (*20/32)
                loss5, _ = loss_interp(pr5, pr5_input, pr5_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_5, deltaWeights)
                upconv4 = slim.conv2d_transpose(concat5, 256, [2*scale, 2*scale], stride=scale, scope='upconv4')
                pr5to4 = slim.conv2d_transpose(pr5, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr5to4')
                concat4 = tf.concat(3, [end_points["Mixed_5d"], upconv4, pr5to4])

                pr4 = slim.conv2d(concat4, 2, [3, 3], activation_fn=None, scope='pr4')
                h4 = pr4.get_shape()[1].value
                w4 = pr4.get_shape()[2].value
                pr4_input = tf.image.resize_bilinear(inputs_norm, [h4, w4])
                pr4_output = tf.image.resize_bilinear(outputs_norm, [h4, w4])
                flow_scale_4 = 2.5    # (*20/16)
                loss4, _ = loss_interp(pr4, pr4_input, pr4_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_4, deltaWeights)
                scale = 1
                upconv3 = slim.conv2d_transpose(concat4, 128, [2*scale, 2*scale], stride=scale, scope='upconv3')
                pr4to3 = slim.conv2d_transpose(pr4, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr4to3')
                concat3 = tf.concat(3, [end_points["MaxPool_5a_3x3"], upconv3, pr4to3])

                scale = 2
                pr3 = slim.conv2d(concat3, 2, [3, 3], activation_fn=None, scope='pr3')
                h3 = pr3.get_shape()[1].value
                w3 = pr3.get_shape()[2].value
                pr3_input = tf.image.resize_bilinear(inputs_norm, [h3, w3])
                pr3_output = tf.image.resize_bilinear(outputs_norm, [h3, w3])
                flow_scale_3 = 2.5    # (*20/8)
                loss3, _ = loss_interp(pr3, pr3_input, pr3_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_3, deltaWeights)
                upconv2 = slim.conv2d_transpose(concat3, 64, [2*scale, 2*scale], stride=scale, scope='upconv2')
                pr3to2 = slim.conv2d_transpose(pr3, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr3to2')
                concat2 = tf.concat(3, [end_points["MaxPool_3a_3x3"], upconv2, pr3to2])

                pr2 = slim.conv2d(concat2, 2, [3, 3], activation_fn=None, scope='pr2')
                h2 = pr2.get_shape()[1].value
                w2 = pr2.get_shape()[2].value
                pr2_input = tf.image.resize_bilinear(inputs_norm, [h2, w2])
                pr2_output = tf.image.resize_bilinear(outputs_norm, [h2, w2])
                flow_scale_2 = 5.0    # (*20/4)
                loss2, _ = loss_interp(pr2, pr2_input, pr2_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_2, deltaWeights)
                upconv1 = slim.conv2d_transpose(concat2, 32, [2*scale, 2*scale], stride=scale, scope='upconv1')
                pr2to1 = slim.conv2d_transpose(pr2, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr2to1')
                concat1 = tf.concat(3, [end_points["Conv2d_1a_3x3"], upconv1, pr2to1])

                pr1 = slim.conv2d(concat1, 2, [3, 3], activation_fn=None, scope='pr1')
                h1 = pr1.get_shape()[1].value
                w1 = pr1.get_shape()[2].value
                pr1_input = tf.image.resize_bilinear(inputs_norm, [h1, w1])
                pr1_output = tf.image.resize_bilinear(outputs_norm, [h1, w1])
                flow_scale_1 = 10.0    # (*20/2) 
                loss1, prev1 = loss_interp(pr1, pr1_input, pr1_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_1, deltaWeights)
                
                # Adding intermediate losses
                all_loss = loss_weight[0]*loss1["total"] + loss_weight[1]*loss2["total"] + loss_weight[2]*loss3["total"] + \
                            loss_weight[3]*loss4["total"] + loss_weight[4]*loss5["total"] + loss_weight[5]*loss6["total"]
                slim.losses.add_loss(all_loss)

                losses = [loss1, loss2, loss3, loss4, loss5, loss6]
                flows_all = [pr1*flow_scale_1, pr2*flow_scale_2, pr3*flow_scale_3, pr4*flow_scale_4, pr5*flow_scale_5, pr6*flow_scale_6]
                # flows_all = [pr1, pr2, pr3, pr4, pr5, pr6]
                return losses, flows_all, prev1


def inception_v3_arg_scope(weight_decay=0.00004,
                           batch_norm_var_collection='moving_vars'):
    """Defines the default InceptionV3 arg scope.
    Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.
    Returns:
    An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
        }
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.elu,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as sc:
            return sc