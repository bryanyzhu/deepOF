import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def flowNet(inputs, outputs, loss_weight):
    """Creates a warp flow model based on flowNet simple.

    Args:
    inputs: 4D image tensor corresponding to prev frames
    outputs: 4D image tensor corresponding to next frames

    Returns:
    predicted next frames
    """
    # Mean subtraction (BGR) for flying chairs
    mean = tf.constant([97.533268117955444, 99.238235788550085, 97.055973199626948], dtype=tf.float32, name="img_global_mean")
    # tf.tile(mean, [4,192,256,1])
    inputs = inputs - mean
    outputs = outputs - mean
    # Scaling to 0 ~ 1 or -0.4 ~ 0.6?
    inputs = tf.truediv(inputs, 255.0)
    outputs = tf.truediv(outputs, 255.0)

    # Add local response normalization (ACROSS_CHANNELS) for computing photometric loss
    inputs_norm = tf.nn.local_response_normalization(inputs, depth_radius=4, beta=0.7)
    outputs_norm = tf.nn.local_response_normalization(outputs, depth_radius=4, beta=0.7)

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
                        activation_fn=tf.nn.elu):       # original use leaky ReLU, now we use elu
        # Contracting part
        conv1   = slim.conv2d(tf.concat(3, [inputs, outputs]), 64, [7, 7], stride=2, scope='conv1')
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
        epsilon = 0.0001 
        alpha_c = 0.25
        alpha_s = 0.37
        lambda_smooth = 1.0
        deltaWeights = {}
        deltaWeights["FlowDeltaWeights"] = tf.constant([0,0,0,0,1,-1,0,0,0,0,0,0,0,1,0,0,-1,0], dtype=tf.float32, shape=[3,3,2,2], name="FlowDeltaWeights")
        deltaWeights["ImgDeltaWeights"] = tf.constant([-1,0,1,-2,0,2,-1,0,1,-1,-2,-1,0,0,0,1,2,1], dtype=tf.float32, shape=[3,3,1,2], name="ImgDeltaWeights")
        #deltaWeights["xDeltaFlow"] = tf.constant([0,0,0,0,1,0,0,-1,0], dtype=tf.float32, shape=[3,3,2,2], name="xDeltaFlow")
        #deltaWeights["yDeltaFlow"] = tf.constant([0,0,0,0,1,-1,0,0,0], dtype=tf.float32, shape=[3,3,2,2], name="yDeltaFlow")
        #deltaWeights["xDeltaImg"]  = tf.constant([-1,-2,-1,0,0,0,1,2,1], dtype=tf.float32, shape=[3,3,1,1], name="xDeltaImg")
        #deltaWeights["yDeltaImg"]  = tf.constant([-1,0,1,-2,0,2,-1,0,1], dtype=tf.float32, shape=[3,3,1,1], name="yDeltaImg")
        deltaWeights["mean"] = mean
        scale = 2       # for deconvolution

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

def VGG16(inputs, outputs, loss_weight):
    
    # Mean subtraction (BGR) for flying chairs
    mean = tf.constant([97.533268117955444, 99.238235788550085, 97.055973199626948], dtype=tf.float32, name="img_global_mean")
    # tf.tile(mean, [4,192,256,1])
    inputs = inputs - mean
    outputs = outputs - mean
    # Scaling to 0 ~ 1 or -0.4 ~ 0.6?
    inputs = tf.truediv(inputs, 255.0)
    outputs = tf.truediv(outputs, 255.0)

    # Add local response normalization (ACROSS_CHANNELS) for computing photometric loss
    inputs_norm = tf.nn.local_response_normalization(inputs, depth_radius=4, beta=0.7)
    outputs_norm = tf.nn.local_response_normalization(outputs, depth_radius=4, beta=0.7)

    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose], 
                        activation_fn=tf.nn.elu):       # original use leaky ReLU, now we use elu
        
        conv1_1 = slim.conv2d(tf.concat(3, [inputs, outputs]), 64, [3, 3], scope='conv1_1')
        # conv1_1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
        conv1_2 = slim.conv2d(conv1_1, 64, [3, 3], scope='conv1_2')
        pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

        conv2_1 = slim.conv2d(pool1, 128, [3, 3], scope='conv2_1')
        conv2_2 = slim.conv2d(conv2_1, 128, [3, 3], scope='conv2_2')
        pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

        conv3_1 = slim.conv2d(pool2, 256, [3, 3], scope='conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 256, [3, 3], scope='conv3_2')
        conv3_3 = slim.conv2d(conv3_2, 256, [3, 3], scope='conv3_3')
        pool3 = slim.max_pool2d(conv3_3, [2, 2], scope='pool3')

        conv4_1 = slim.conv2d(pool3, 512, [3, 3], scope='conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 512, [3, 3], scope='conv4_2')
        conv4_3 = slim.conv2d(conv4_2, 512, [3, 3], scope='conv4_3')
        pool4 = slim.max_pool2d(conv4_3, [2, 2], scope='pool4')

        conv5_1 = slim.conv2d(pool4, 512, [3, 3], scope='conv5_1')
        conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], scope='conv5_2')
        conv5_3 = slim.conv2d(conv5_2, 512, [3, 3], scope='conv5_3')
        pool5 = slim.max_pool2d(conv5_3, [2, 2], scope='pool5')
       
        # Hyper-params for computing unsupervised loss
        epsilon = 0.0001 
        alpha_c = 0.25
        alpha_s = 0.37
        lambda_smooth = 1.0
        deltaWeights = {}
        deltaWeights["FlowDeltaWeights"] = tf.constant([0,0,0,0,1,-1,0,0,0,0,0,0,0,1,0,0,-1,0], dtype=tf.float32, shape=[3,3,2,2], name="FlowDeltaWeights")
        # deltaWeights["ImgDeltaWeights"] = tf.constant([-1,0,1,-2,0,2,-1,0,1,-1,-2,-1,0,0,0,1,2,1], dtype=tf.float32, shape=[3,3,1,2], name="ImgDeltaWeights")
        # deltaWeights["mean"] = mean
        scale = 2       # for deconvolution

        # Expanding part
        pr5 = slim.conv2d(pool5, 2, [3, 3], activation_fn=None, scope='pr5')
        h5 = pr5.get_shape()[1].value
        w5 = pr5.get_shape()[2].value
        pr5_input = tf.image.resize_bilinear(inputs_norm, [h5, w5])
        pr5_output = tf.image.resize_bilinear(outputs_norm, [h5, w5])
        flow_scale_5 = 0.625    # (*20/32)
        loss5, _ = loss_interp(pr5, pr5_input, pr5_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_5, deltaWeights)
        upconv4 = slim.conv2d_transpose(pool5, 256, [2*scale, 2*scale], stride=scale, scope='upconv4')
        pr5to4 = slim.conv2d_transpose(pr5, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr5to4')
        concat4 = tf.concat(3, [pool4, upconv4, pr5to4])

        pr4 = slim.conv2d(concat4, 2, [3, 3], activation_fn=None, scope='pr4')
        h4 = pr4.get_shape()[1].value
        w4 = pr4.get_shape()[2].value
        pr4_input = tf.image.resize_bilinear(inputs_norm, [h4, w4])
        pr4_output = tf.image.resize_bilinear(outputs_norm, [h4, w4])
        flow_scale_4 = 1.25    # (*20/16)
        loss4, _ = loss_interp(pr4, pr4_input, pr4_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_4, deltaWeights)
        upconv3 = slim.conv2d_transpose(concat4, 128, [2*scale, 2*scale], stride=scale, scope='upconv3')
        pr4to3 = slim.conv2d_transpose(pr4, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr4to3')
        concat3 = tf.concat(3, [pool3, upconv3, pr4to3])

        pr3 = slim.conv2d(concat3, 2, [3, 3], activation_fn=None, scope='pr3')
        h3 = pr3.get_shape()[1].value
        w3 = pr3.get_shape()[2].value
        pr3_input = tf.image.resize_bilinear(inputs_norm, [h3, w3])
        pr3_output = tf.image.resize_bilinear(outputs_norm, [h3, w3])
        flow_scale_3 = 2.5    # (*20/8)
        loss3, _ = loss_interp(pr3, pr3_input, pr3_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_3, deltaWeights)
        upconv2 = slim.conv2d_transpose(concat3, 64, [2*scale, 2*scale], stride=scale, scope='upconv2')
        pr3to2 = slim.conv2d_transpose(pr3, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr3to2')
        concat2 = tf.concat(3, [pool2, upconv2, pr3to2])

        pr2 = slim.conv2d(concat2, 2, [3, 3], activation_fn=None, scope='pr2')
        h2 = pr2.get_shape()[1].value
        w2 = pr2.get_shape()[2].value
        pr2_input = tf.image.resize_bilinear(inputs_norm, [h2, w2])
        pr2_output = tf.image.resize_bilinear(outputs_norm, [h2, w2])
        flow_scale_2 = 5.0    # (*20/4)
        loss2, _ = loss_interp(pr2, pr2_input, pr2_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_2, deltaWeights)
        upconv1 = slim.conv2d_transpose(concat2, 32, [2*scale, 2*scale], stride=scale, scope='upconv1')
        pr2to1 = slim.conv2d_transpose(pr2, 2, [2*scale, 2*scale], stride=scale, activation_fn=None, scope='up_pr2to1')
        concat1 = tf.concat(3, [pool1, upconv1, pr2to1])

        pr1 = slim.conv2d(concat1, 2, [3, 3], activation_fn=None, scope='pr1')
        h1 = pr1.get_shape()[1].value
        w1 = pr1.get_shape()[2].value
        pr1_input = tf.image.resize_bilinear(inputs_norm, [h1, w1])
        pr1_output = tf.image.resize_bilinear(outputs_norm, [h1, w1])
        flow_scale_1 = 10.0    # (*20/2) 
        loss1, prev1 = loss_interp(pr1, pr1_input, pr1_output, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale_1, deltaWeights)
        
        # Adding intermediate losses
        all_loss = loss_weight[0]*loss1["total"] + loss_weight[1]*loss2["total"] + loss_weight[2]*loss3["total"] + \
                    loss_weight[3]*loss4["total"] + loss_weight[4]*loss5["total"] 
        slim.losses.add_loss(all_loss)

        losses = [loss1, loss2, loss3, loss4, loss5, loss5]
        flows_all = [pr1*flow_scale_1, pr2*flow_scale_2, pr3*flow_scale_3, pr4*flow_scale_4, pr5*flow_scale_5, pr5*flow_scale_5]
        
        return losses, flows_all, prev1


def loss_interp(flows, inputs, outputs, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale, deltaWeights):
# def loss_interp(flows, outputs, inputs, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale, deltaWeights):

    shape = inputs.get_shape()
    shape = [int(dim) for dim in shape]
    num_batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    needMask = True
    # Create border mask for image
    border_ratio = 0.1
    shortestDim = height
    borderWidth = int(np.ceil(shortestDim * border_ratio))
    smallerMask = tf.ones([height-2*borderWidth, width-2*borderWidth])
    borderMask = tf.pad(smallerMask, [[borderWidth,borderWidth], [borderWidth,borderWidth]], "CONSTANT")
    borderMask = tf.tile(tf.expand_dims(borderMask, 0), [num_batch, 1, 1])
    borderMaskImg = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, channels])
    borderMaskFlow = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, 2])

    # Create smoothness border mask for optical flow
    smallerSmoothMaskx = tf.ones([height-1, width])
    smallerSmoothMasky = tf.ones([height, width-1])
    smoothnessMaskx = tf.pad(smallerSmoothMaskx, [[0,1], [0,0]], "CONSTANT")    # vertical
    smoothnessMasky = tf.pad(smallerSmoothMasky, [[0,0], [0,1]], "CONSTANT")    # horizontal
    smoothnessMask = tf.pack([smoothnessMasky, smoothnessMaskx], axis=2)
    smoothnessMask = tf.tile(tf.expand_dims(smoothnessMask, 0), [num_batch, 1, 1, 1])

    inputs_flat = tf.reshape(inputs, [num_batch, -1, channels])
    outputs_flat = tf.reshape(outputs, [num_batch, -1, channels])
    borderMask_flat = tf.reshape(borderMaskImg, [num_batch, -1, channels])

    flows = tf.mul(flows, flow_scale)
    flows_flat = tf.reshape(flows, [num_batch, -1, 2])
    floor_flows = tf.to_int32(tf.floor(flows_flat))
    weights_flows = flows_flat - tf.floor(flows_flat)

    # Construct the grids
    pos_x = tf.range(height)
    pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, width])
    pos_x = tf.reshape(pos_x, [-1])
    pos_y = tf.range(width)
    pos_y = tf.tile(tf.expand_dims(pos_y, 0), [height, 1])
    pos_y = tf.reshape(pos_y, [-1])
    zero = tf.zeros([], dtype='int32')

    # Warp two images based on optical flow
    batch = []
    for b in range(num_batch):
        channel = []
        x = floor_flows[b, :, 0]    # U, horizontal displacement
        y = floor_flows[b, :, 1]    # V, vertical displacement
        xw = weights_flows[b, :, 0]
        yw = weights_flows[b, :, 1]

        for c in range(channels):

            x0 = pos_y + x
            x1 = x0 + 1
            y0 = pos_x + y
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, width-1)
            x1 = tf.clip_by_value(x1, zero, width-1)
            y0 = tf.clip_by_value(y0, zero, height-1)
            y1 = tf.clip_by_value(y1, zero, height-1)

            idx_a = y0 * width + x0
            idx_b = y1 * width + x0
            idx_c = y0 * width + x1
            idx_d = y1 * width + x1

            Ia = tf.gather(outputs_flat[b, :, c], idx_a)
            Ib = tf.gather(outputs_flat[b, :, c], idx_b)
            Ic = tf.gather(outputs_flat[b, :, c], idx_c)
            Id = tf.gather(outputs_flat[b, :, c], idx_d)

            wa = (1-xw) * (1-yw)
            wb = (1-xw) * yw
            wc = xw * (1-yw)
            wd = xw * yw

            img = tf.mul(Ia, wa) + tf.mul(Ib, wb) + tf.mul(Ic, wc) + tf.mul(Id, wd)
            channel.append(img)
        batch.append(tf.pack(channel, axis=1))
    reconstructs = tf.pack(batch)
    
    # Recostruction loss
    diff_reconstruct = tf.scalar_mul(255.0, tf.sub(reconstructs, inputs_flat))
    eleWiseLoss = tf.pow(tf.square(diff_reconstruct) + tf.square(epsilon), alpha_c)
    Charbonnier_reconstruct = 0.0
    numValidPixels = 0.0
    if needMask:
        eleWiseLoss = tf.mul(borderMask_flat, eleWiseLoss)
        validPixels = tf.equal(borderMask_flat, tf.ones_like(borderMask_flat))
        numValidPixels = tf.to_float(tf.reduce_sum(tf.to_int32(validPixels)))
        Charbonnier_reconstruct = tf.reduce_sum(eleWiseLoss) / numValidPixels
    else:
        Charbonnier_reconstruct = tf.reduce_mean(eleWiseLoss)

    # Smoothness loss
    flow_delta = tf.nn.conv2d(flows, deltaWeights["FlowDeltaWeights"], [1,1,1,1], padding="SAME")  
    
    U_loss = 0.0
    V_loss = 0.0
    if needMask:
        flow_delta_clean = flow_delta * smoothnessMask
        eleWiseULoss = tf.pow(tf.square(flow_delta_clean[:,:,:,0]) + tf.square(epsilon), alpha_s)
        U_loss = tf.reduce_sum(eleWiseULoss) / numValidPixels
        eleWiseVLoss = tf.pow(tf.square(flow_delta_clean[:,:,:,1]) + tf.square(epsilon), alpha_s)
        V_loss = tf.reduce_sum(eleWiseVLoss) / numValidPixels
    else:
        U_loss = tf.reduce_mean(tf.pow(tf.square(flow_delta[:,:,:,0])  + tf.square(epsilon), alpha_s)) 
        V_loss = tf.reduce_mean(tf.pow(tf.square(flow_delta[:,:,:,1])  + tf.square(epsilon), alpha_s))
    loss_smooth = U_loss + V_loss

    total_loss = Charbonnier_reconstruct + lambda_smooth * loss_smooth
    # Define a loss structure
    lossDict = {}
    lossDict["total"] = total_loss
    lossDict["Charbonnier_reconstruct"] = Charbonnier_reconstruct
    lossDict["U_loss"] = U_loss
    lossDict["V_loss"] = V_loss
    return lossDict, tf.reshape(reconstructs, [num_batch, height, width, 3])

def loss_interp_bk(flows, inputs, outputs, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale, deltaWeights):

    shape = inputs.get_shape()
    shape = [int(dim) for dim in shape]
    num_batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    needMask = True
    # Create border mask for image
    border_ratio = 0.1
    shortestDim = height
    borderWidth = int(np.ceil(shortestDim * border_ratio))
    smallerMask = tf.ones([height-2*borderWidth, width-2*borderWidth])
    borderMask = tf.pad(smallerMask, [[borderWidth,borderWidth], [borderWidth,borderWidth]], "CONSTANT")
    borderMask = tf.tile(tf.expand_dims(borderMask, 0), [num_batch, 1, 1])
    borderMaskImg = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, channels])
    borderMaskFlow = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, 2])

    # Create smoothness border mask for optical flow
    smallerSmoothMaskx = tf.ones([height-1, width])
    smallerSmoothMasky = tf.ones([height, width-1])
    smoothnessMaskx = tf.pad(smallerSmoothMaskx, [[0,1], [0,0]], "CONSTANT")    # vertical
    smoothnessMasky = tf.pad(smallerSmoothMasky, [[0,0], [0,1]], "CONSTANT")    # horizontal
    smoothnessMask = tf.pack([smoothnessMasky, smoothnessMaskx], axis=2)
    smoothnessMask = tf.tile(tf.expand_dims(smoothnessMask, 0), [num_batch, 1, 1, 1])

    inputs_flat = tf.reshape(inputs, [num_batch, -1, channels])
    outputs_flat = tf.reshape(outputs, [num_batch, -1, channels])
    borderMask_flat = tf.reshape(borderMaskImg, [num_batch, -1, channels])

    flows = tf.mul(flows, flow_scale)
    flows_flat = tf.reshape(flows, [num_batch, -1, 2])
    floor_flows = tf.to_int32(tf.floor(flows_flat))
    weights_flows = flows_flat - tf.floor(flows_flat)

    # Construct the grids
    pos_x = tf.range(height)
    pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, width])
    pos_x = tf.reshape(pos_x, [-1])
    pos_y = tf.range(width)
    pos_y = tf.tile(tf.expand_dims(pos_y, 0), [height, 1])
    pos_y = tf.reshape(pos_y, [-1])
    zero = tf.zeros([], dtype='int32')

    # Warp two images based on optical flow
    batch = []
    for b in range(num_batch):
        channel = []
        x = floor_flows[b, :, 0]    # U, horizontal displacement
        y = floor_flows[b, :, 1]    # V, vertical displacement
        xw = weights_flows[b, :, 0]
        yw = weights_flows[b, :, 1]

        for c in range(channels):

            x0 = pos_y + x
            x1 = x0 + 1
            y0 = pos_x + y
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, width-1)
            x1 = tf.clip_by_value(x1, zero, width-1)
            y0 = tf.clip_by_value(y0, zero, height-1)
            y1 = tf.clip_by_value(y1, zero, height-1)

            idx_a = y0 * width + x0
            idx_b = y1 * width + x0
            idx_c = y0 * width + x1
            idx_d = y1 * width + x1

            Ia = tf.gather(outputs_flat[b, :, c], idx_a)
            Ib = tf.gather(outputs_flat[b, :, c], idx_b)
            Ic = tf.gather(outputs_flat[b, :, c], idx_c)
            Id = tf.gather(outputs_flat[b, :, c], idx_d)

            wa = (1-xw) * (1-yw)
            wb = (1-xw) * yw
            wc = xw * (1-yw)
            wd = xw * yw

            img = tf.mul(Ia, wa) + tf.mul(Ib, wb) + tf.mul(Ic, wc) + tf.mul(Id, wd)
            channel.append(img)
        batch.append(tf.pack(channel, axis=1))
    reconstructs = tf.pack(batch)
    
    # Recostruction loss
    diff_reconstruct = tf.scalar_mul(255.0, tf.sub(reconstructs, inputs_flat))
    eleWiseLoss = tf.pow(tf.square(diff_reconstruct) + tf.square(epsilon), alpha_c)
    Charbonnier_reconstruct = 0.0
    numValidPixels = 0.0
    if needMask:
        eleWiseLoss = tf.mul(borderMask_flat, eleWiseLoss)
        validPixels = tf.equal(borderMask_flat, tf.ones_like(borderMask_flat))
        numValidPixels = tf.to_float(tf.reduce_sum(tf.to_int32(validPixels)))
        Charbonnier_reconstruct = tf.reduce_sum(eleWiseLoss) / numValidPixels
    else:
        Charbonnier_reconstruct = tf.reduce_mean(eleWiseLoss)

    # Smoothness loss
    flow_delta = tf.nn.conv2d(flows, deltaWeights["FlowDeltaWeights"], [1,1,1,1], padding="SAME")    # vertical
    #U_delta = flow_delta[:,:,:,0:2]
    #V_delta = flow_delta[:,:,:,2:4]
    #flow_delta_x = tf.nn.conv2d(flows, deltaWeights["xDeltaFlow"], [1,1,1,1], padding="SAME")    # vertical
    #flow_delta_y = tf.nn.conv2d(flows, deltaWeights["yDeltaFlow"], [1,1,1,1], padding="SAME")    # horizontal

    # Add image gradients, edge-aware smoothness
    img_delta = tf.scalar_mul(255.0, inputs) + deltaWeights["mean"]
    img_delta_clip = tf.clip_by_value(tf.to_int32(img_delta), zero, 255)
    inputs_gray = tf.to_float(tf.image.rgb_to_grayscale(img_delta_clip))
    img_delta = tf.nn.conv2d(inputs_gray, deltaWeights["ImgDeltaWeights"], [1,1,1,1], padding="SAME")
    #img_delta_x = tf.nn.conv2d(inputs_gray, deltaWeights["xDeltaImg"], [1,1,1,1], padding="SAME")    # vertical
    #img_delta_y = tf.nn.conv2d(inputs_gray, deltaWeights["yDeltaImg"], [1,1,1,1], padding="SAME")    # horizontal

    #U_delta = tf.pack([flow_delta_x[:,:,:,0], flow_delta_y[:,:,:,0]], axis=3)
    #V_delta = tf.pack([flow_delta_x[:,:,:,1], flow_delta_y[:,:,:,1]], axis=3)
    
    U_loss = 0.0
    V_loss = 0.0
    # not the same number of valid pixels for images and optical flow, 2/3 relation, channel is different
    if needMask:
        flow_delta_clean = flow_delta * smoothnessMask
        eleWiseULoss = tf.pow(tf.square(flow_delta_clean[:,:,:,0]) + tf.square(epsilon), alpha_s)
        img_U_weight = tf.exp(-1.0 * tf.abs(img_delta[:,:,:,0]))
        U_loss = tf.reduce_sum(eleWiseULoss * img_U_weight) / (numValidPixels / 3 * 2)
        eleWiseVLoss = tf.pow(tf.square(flow_delta_clean[:,:,:,1]) + tf.square(epsilon), alpha_s)
        img_V_weight = tf.exp(-1.0 * tf.abs(img_delta[:,:,:,1]))
        V_loss = tf.reduce_sum(eleWiseVLoss * img_V_weight) / (numValidPixels / 3 * 2)

        # Ux = tf.pow(tf.square(U_delta_border_clean[:,:,:,0]) + tf.square(epsilon), alpha_s)
        # Uy = tf.pow(tf.square(U_delta_border_clean[:,:,:,1]) + tf.square(epsilon), alpha_s)
        # #Ux = tf.mul(Ux, tf.exp(-1.0 * tf.abs(tf.squeeze(img_delta_x))))
        # #Uy = tf.mul(Uy, tf.exp(-1.0 * tf.abs(tf.squeeze(img_delta_y))))
        # # U_eleWiseLoss = tf.mul(U_eleWiseLoss, tf.exp(-1.0 * img_delta_x))
        # U_loss = tf.reduce_sum(tf.pack([Ux, Uy], axis=3)) / (numValidPixels / 3 * 2)

        # Vx = tf.pow(tf.square(V_delta_border_clean[:,:,:,0]) + tf.square(epsilon), alpha_s)
        # Vy = tf.pow(tf.square(V_delta_border_clean[:,:,:,1]) + tf.square(epsilon), alpha_s)
        # #Vx = tf.mul(Vx, tf.exp(-1.0 * tf.abs(tf.squeeze(img_delta_x))))
        # #Vy = tf.mul(Vy, tf.exp(-1.0 * tf.abs(tf.squeeze(img_delta_y))))
        # #U_eleWiseLoss = tf.mul(U_eleWiseLoss, tf.exp(-1.0 * img_delta_x))
        # V_loss = tf.reduce_sum(tf.pack([Vx, Vy], axis=3)) / (numValidPixels / 3 * 2)
    else:
        U_loss = tf.reduce_mean(tf.pow(tf.square(flow_delta[:,:,:,0] * flow_scale)  + tf.square(epsilon), alpha_s)) 
        V_loss = tf.reduce_mean(tf.pow(tf.square(flow_delta[:,:,:,1] * flow_scale)  + tf.square(epsilon), alpha_s))
    loss_smooth = U_loss + V_loss

    total_loss = Charbonnier_reconstruct + lambda_smooth * loss_smooth
    # Define a loss structure
    lossDict = {}
    lossDict["total"] = total_loss
    lossDict["Charbonnier_reconstruct"] = Charbonnier_reconstruct
    lossDict["U_loss"] = U_loss
    lossDict["V_loss"] = V_loss
    return lossDict, tf.reshape(reconstructs, [num_batch, height, width, 3])