import os,sys
import numpy as np
import tensorflow as tf
import logging
from math import ceil

class deep3D:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.parameters = []
        # Main branch: VGG16
        self.conv_layers()
        # # Initialize deep3D with VGG pre-trained network params
        # if weights is not None and sess is not None:
        #     self.load_weights(weights, sess)
        # fc layers
        self.fc_layers()
        # side branches including batch normalization and convolution
        self.side_branches()
        # Deconvolution
        self.deconv_layers()
    

    def conv_layers(self):
        # TODO: zero-mean input
        images = self.imgs

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):

    	# fc6
    	with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc6w = tf.Variable(tf.truncated_normal([shape, 512],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc6b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.relu(fc6l)
            self.drop6 = tf.nn.dropout(self.fc6, keep_prob=0.5)
            self.parameters += [fc6w, fc6b]

        # fc7
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(tf.truncated_normal([512, 512],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc7b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            
            fc7l = tf.nn.bias_add(tf.matmul(self.drop6, fc7w), fc7b)
            self.fc7 = tf.nn.relu(fc7l)
            self.drop7 = tf.nn.dropout(self.fc7, keep_prob=0.5)
            self.parameters += [fc7w, fc7b]

        # fc8, output layer and reshape
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(tf.truncated_normal([512, 33*14*32],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc8b = tf.Variable(tf.constant(1.0, shape=[33*14*32], dtype=tf.float32),
                                 trainable=True, name='biases')
            
            self.fc8l = tf.nn.bias_add(tf.matmul(self.drop7, fc8w), fc8b)
            self.parameters += [fc8w, fc8b]
            

    def side_branches(self):

    	# pred5
    	self.pred5 = tf.reshape(self.fc8l, [1, 14, 32, 33])

    	# pred4
    	# bn_pool4 = tf.nn.batch_normalization(self.pool4)
    	self.bn_pool4 = tf.contrib.layers.batch_norm(self.pool4)
    	with tf.name_scope('pred4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 33], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.bn_pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[33], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.pred4 = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]

        # pred3
    	self.bn_pool3 = tf.contrib.layers.batch_norm(self.pool3)
    	with tf.name_scope('pred3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 33], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.bn_pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[33], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.pred3 = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]

        # pred2
    	self.bn_pool2 = tf.contrib.layers.batch_norm(self.pool2)
    	with tf.name_scope('pred2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 33], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.bn_pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[33], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.pred2 = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]

        # pred1
    	self.bn_pool1 = tf.contrib.layers.batch_norm(self.pool1)
    	with tf.name_scope('pred1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 33], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.bn_pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[33], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.pred1 = tf.nn.bias_add(conv, biases)
            self.parameters += [kernel, biases]

    def deconv_layers(self):
    	debug = True
        print self.pred1.get_shape()
        print self.pred2.get_shape()
        print self.pred3.get_shape()
        print self.pred4.get_shape()
        print self.pred5.get_shape()
        
    	# pred1_1
    	scale = 1
    	self.pred1 = tf.nn.relu(self.pred1)
    	self.pred1_1 = self.upscore_layer(self.pred1,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='pred1_1',
                                            ksize=scale, stride=scale, pad=0)
        
        # pred2_1
        scale *= 2
        self.pred2 = tf.nn.relu(self.pred2)
        self.pred2_1 = self.upscore_layer(self.pred2,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='pred2_1',
                                            ksize=2*scale, stride=scale, pad=scale/2)
        
        # pred3_1
        scale *= 2
        self.pred3 = tf.nn.relu(self.pred3)
        self.pred3_1 = self.upscore_layer(self.pred3,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='pred3_1',
                                            ksize=2*scale, stride=scale, pad=scale/2)

        # pred4_1
        scale *= 2
        self.pred4 = tf.nn.relu(self.pred4)
        self.pred4_1 = self.upscore_layer(self.pred4,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='pred4_1',
                                            ksize=2*scale, stride=scale, pad=scale/2)

        # pred5_1
        scale *= 2
        self.pred5 = tf.nn.relu(self.pred5)
        self.pred5_1 = self.upscore_layer(self.pred5,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='pred5_1',
                                            ksize=2*scale, stride=scale, pad=scale/2)

        # feat
        feat = tf.add_n([self.pred1_1, self.pred2_1, self.pred3_1, self.pred4_1, self.pred5_1])
        self.feat_act = tf.nn.relu(feat)
        scale = 2
        up = self.upscore_layer(self.feat_act,
                                            shape=[1,218,512],
                                            num_classes=33,
                                            debug=debug, name='feat',
                                            ksize=2*scale, stride=scale, pad=scale/2)

        self.up_act= tf.nn.relu(up)
        # conv_feat
        with tf.name_scope('conv_feat') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 33, 33], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.up_act, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[33], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.conv_feat = tf.nn.bias_add(conv, biases)
            

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        cutLayerNum = [2,3,6,7,12,13,18,19,24,25]
        offNum = 0
        for i, k in enumerate(keys):
        	if i <= 25:
	        	if i in cutLayerNum:
	        		offNum += 1
	        		print i, k, np.shape(weights[k]), "not included in deep3D model"
	        	else:
	        		print i, k, np.shape(weights[k])
	        		sess.run(self.parameters[i-offNum].assign(weights[k]))

    def upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize, stride, pad):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            kernel_h = kernel_w = ksize
            stride_h = stride_w = stride
            pad_h = pad_w = pad
            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)
                # in_shape = bottom.get_shape()
                # TODO: fix shape problem if output_shape is not specified
                h = ((in_shape[1] - 1) * stride_h) + kernel_h - 2 * pad_h
                w = ((in_shape[2] - 1) * stride_w) + kernel_w - 2 * pad_w
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            
            # print new_shape
            output_shape = tf.pack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)
	            
