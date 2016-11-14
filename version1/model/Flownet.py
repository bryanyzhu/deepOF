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

def model(source_imgs, target_imgs, sample_mean, loss_weight, hyper_params, is_training):
    """Creates a warp flow model based on flowNet simple.

    Args:
    inputs: 4D image tensor corresponding to prev frames
    outputs: 4D image tensor corresponding to next frames

    Returns:
    predicted next frames
    """
    # Global mean value of training dataset in BGR order 
    mean = sample_mean

    # Pre-scaling, for both train and test
    inputs = tf.truediv(tf.sub(source_imgs, mean), 255.0)
    outputs = tf.truediv(tf.sub(target_imgs, mean), 255.0)
    # Augmentation for training
    geo_in, geo_out, photo_in, photo_out = tf.cond(is_training, 
        lambda: augmentation(inputs, outputs, mean), lambda: keepOrigin(inputs, outputs))

    # Add local response normalization (ACROSS_CHANNELS) for computing photometric loss
    inputs_norm = tf.nn.local_response_normalization(geo_in, depth_radius=4, beta=0.7)
    outputs_norm = tf.nn.local_response_normalization(geo_out, depth_radius=4, beta=0.7)

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                        weights_initializer=initializers.xavier_initializer(),
                        weights_regularizer=None,
                        biases_initializer=init_ops.zeros_initializer,
                        biases_regularizer=None,
                        activation_fn=tf.nn.elu):       # original use leaky ReLU, now we use elu
        # Contracting part
        conv1   = slim.conv2d(tf.concat(3, [photo_in, photo_out]), 64, [7, 7], stride=2, scope='conv1')
        conv2   = slim.conv2d(conv1, 128, [5, 5], stride=2, scope='conv2')
        conv3_1 = slim.conv2d(conv2, 256, [5, 5], stride=2, scope='conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 256, [3, 3], scope='conv3_2')
        conv4_1 = slim.conv2d(conv3_2, 512, [3, 3], stride=2, scope='conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 512, [3, 3], scope='conv4_2')
        conv5_1 = slim.conv2d(conv4_2, 512, [3, 3], stride=2, scope='conv5_1')
        conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], scope='conv5_2')
        conv6_1 = slim.conv2d(conv5_2, 1024, [3, 3], stride=2, scope='conv6_1')
        conv6_2 = slim.conv2d(conv6_1, 1024, [3, 3], scope='conv6_2')

        # Hyper-params for computing unsupervised loss
        lambda_smooth = hyper_params[0]
        epsilon = hyper_params[1]
        alpha_c = hyper_params[2]
        alpha_s = hyper_params[3]
        scale = 2       # for deconvolution
        
        deltaWeights = {}
        needMask = True
        deltaWeights["needMask"] = needMask
        # Calculating flow derivatives
        flow_width = tf.constant([[0, 0, 0], [0, 1, -1], [0, 0, 0]], tf.float32)
        flow_width_filter = tf.reshape(flow_width, [3, 3, 1, 1])
        flow_width_filter = tf.tile(flow_width_filter, [1, 1, 2, 1])
        flow_height = tf.constant([[0, 0, 0], [0, 1, 0], [0, -1, 0]], tf.float32)
        flow_height_filter = tf.reshape(flow_height, [3, 3, 1, 1])
        flow_height_filter = tf.tile(flow_height_filter, [1, 1, 2, 1])
        deltaWeights["flow_width_filter"] = flow_width_filter
        deltaWeights["flow_height_filter"] = flow_height_filter

        needImageGradients = False
        deltaWeights["needImageGradients"] = needImageGradients
        if needImageGradients:
            # Calculating image derivatives
            sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
            sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
            deltaWeights["sobel_x_filter"] = sobel_x_filter
            deltaWeights["sobel_y_filter"] = sobel_y_filter

        # Expanding part
        pr6 = slim.conv2d(conv6_2, 2, [3, 3], activation_fn=None, scope='pr6')
        h6 = pr6.get_shape()[1].value
        w6 = pr6.get_shape()[2].value
        pr6_input = tf.image.resize_bilinear(inputs_norm, [h6, w6])
        pr6_output = tf.image.resize_bilinear(outputs_norm, [h6, w6])
        flow_scale_6 = 0.3125    # (*20/64)
        loss6, _ = loss_interp(pr6, pr6_input, pr6_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_6, deltaWeights)
        upconv5 = slim.conv2d_transpose(conv6_2, 512, [2*scale, 2*scale], stride=scale, scope='upconv5')
        pr6to5 = slim.conv2d_transpose(pr6, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr6to5')
        concat5 = tf.concat(3, [conv5_2, upconv5, pr6to5])

        pr5 = slim.conv2d(concat5, 2, [3, 3], activation_fn=None, scope='pr5')
        h5 = pr5.get_shape()[1].value
        w5 = pr5.get_shape()[2].value
        pr5_input = tf.image.resize_bilinear(inputs_norm, [h5, w5])
        pr5_output = tf.image.resize_bilinear(outputs_norm, [h5, w5])
        flow_scale_5 = 0.625    # (*20/32)
        loss5, _ = loss_interp(pr5, pr5_input, pr5_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_5, deltaWeights)
        upconv4 = slim.conv2d_transpose(concat5, 256, [2*scale, 2*scale], stride=scale, scope='upconv4')
        pr5to4 = slim.conv2d_transpose(pr5, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr5to4')
        concat4 = tf.concat(3, [conv4_2, upconv4, pr5to4])

        pr4 = slim.conv2d(concat4, 2, [3, 3], activation_fn=None, scope='pr4')
        h4 = pr4.get_shape()[1].value
        w4 = pr4.get_shape()[2].value
        pr4_input = tf.image.resize_bilinear(inputs_norm, [h4, w4])
        pr4_output = tf.image.resize_bilinear(outputs_norm, [h4, w4])
        flow_scale_4 = 1.25    # (*20/16)
        loss4, _ = loss_interp(pr4, pr4_input, pr4_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_4, deltaWeights)
        upconv3 = slim.conv2d_transpose(concat4, 128, [2*scale, 2*scale], stride=scale, scope='upconv3')
        pr4to3 = slim.conv2d_transpose(pr4, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr4to3')
        concat3 = tf.concat(3, [conv3_2, upconv3, pr4to3])

        pr3 = slim.conv2d(concat3, 2, [3, 3], activation_fn=None, scope='pr3')
        h3 = pr3.get_shape()[1].value
        w3 = pr3.get_shape()[2].value
        pr3_input = tf.image.resize_bilinear(inputs_norm, [h3, w3])
        pr3_output = tf.image.resize_bilinear(outputs_norm, [h3, w3])
        flow_scale_3 = 2.5    # (*20/8)
        loss3, _ = loss_interp(pr3, pr3_input, pr3_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_3, deltaWeights)
        upconv2 = slim.conv2d_transpose(concat3, 64, [2*scale, 2*scale], stride=scale, scope='upconv2')
        pr3to2 = slim.conv2d_transpose(pr3, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr3to2')
        concat2 = tf.concat(3, [conv2, upconv2, pr3to2])

        pr2 = slim.conv2d(concat2, 2, [3, 3], activation_fn=None, scope='pr2')
        h2 = pr2.get_shape()[1].value
        w2 = pr2.get_shape()[2].value
        pr2_input = tf.image.resize_bilinear(inputs_norm, [h2, w2])
        pr2_output = tf.image.resize_bilinear(outputs_norm, [h2, w2])
        flow_scale_2 = 5.0    # (*20/4)
        loss2, _ = loss_interp(pr2, pr2_input, pr2_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_2, deltaWeights)
        upconv1 = slim.conv2d_transpose(concat2, 32, [2*scale, 2*scale], stride=scale, scope='upconv1')
        pr2to1 = slim.conv2d_transpose(pr2, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr2to1')
        concat1 = tf.concat(3, [conv1, upconv1, pr2to1])

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