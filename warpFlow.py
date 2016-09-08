"""Contains a warp flow model, which adapt from vgg16 net.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import logging


def Model(inputs, outputs):
  """Creates the warp flow model.

  Args:
    inputs: 4D image tensor corresponding to prev frames
    outputs: 4D image tensor corresponding to next frames
  Returns:
    predicted next frames
  """

  with slim.arg_scope([slim.conv2d, slim.fully_connected], 
		       activation_fn=tf.nn.relu,
                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                       weights_regularizer=slim.l2_regularizer(0.0005)):

    conv1_1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
    pool1 = slim.max_pool2d(conv1_1, [2, 2], scope='pool1')
    conv2_1 = slim.conv2d(pool1, 64, [3, 3], scope='conv2_1')
    pool2 = slim.max_pool2d(conv2_1, [2, 2], scope='pool2')
    conv3_1 = slim.conv2d(pool2, 64, [3, 3], scope='conv3_1')
    conv3_2 = slim.conv2d(conv3_1, 64, [3, 3], scope='conv3_2')
    pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')
    conv4_1 = slim.conv2d(pool3, 128, [3, 3], scope='conv4_1')
    conv4_2 = slim.conv2d(conv4_1, 128, [3, 3], scope='conv4_2')
    pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')
    conv5_1 = slim.conv2d(pool4, 256, [3, 3], scope='conv5_1')
    conv5_2 = slim.conv2d(conv5_1, 256, [3, 3], scope='conv5_2')
    pool5 = slim.max_pool2d(conv5_2, [2, 2], scope='pool5')
    flatten5 = slim.flatten(pool5, scope='flatten5')
    fc6 = slim.fully_connected(flatten5, 4096, scope='fc6')
    dropout6 = slim.dropout(fc6, 0.5, scope='dropout6')
    fc7 = slim.fully_connected(dropout6, 4096, scope='fc7')
    dropout7 = slim.dropout(fc7, 0.5, scope='dropout7')
    
    channels = 40
 
    fc8 = slim.fully_connected(dropout7, channels*4*5, scope='d5')
    bn_pool1 = slim.batch_norm(pool1, scope='bn_pool1')
    bn_pool2 = slim.batch_norm(pool2, scope='bn_pool2')
    bn_pool3 = slim.batch_norm(pool3, scope='bn_pool3')
    bn_pool4 = slim.batch_norm(pool4, scope='bn_pool4')

    d1 = slim.conv2d(bn_pool1, channels, [3, 3])
    d2 = slim.conv2d(bn_pool2, channels, [3, 3])
    d3 = slim.conv2d(bn_pool3, channels, [3, 3])
    d4 = slim.conv2d(bn_pool4, channels, [3, 3])
    d5 = tf.reshape(fc8, [-1, 4, 5, channels])

    scale = 1
    d1 = slim.conv2d_transpose(d1, channels, [2*scale, 2*scale], stride=scale)
    scale *= 2
    d2 = slim.conv2d_transpose(d2, channels, [2*scale, 2*scale], stride=scale)
    scale *= 2
    d3 = slim.conv2d_transpose(d3, channels, [2*scale, 2*scale], stride=scale)
    scale *= 2
    d4 = slim.conv2d_transpose(d4, channels, [2*scale, 2*scale], stride=scale)
    scale *= 2
    d5 = slim.conv2d_transpose(d5, channels, [2*scale, 2*scale], stride=scale)
    flows = tf.add_n([d1, d2, d3, d4, d5])

    scale = 2
    flows = slim.conv2d_transpose(flows, channels, [2*scale, 2*scale], stride=scale)
    flows = slim.conv2d(flows, 2, [3, 3], activation_fn=None)

    tf.image_summary('Flow-x Map', tf.to_float(tf.expand_dims(tf.expand_dims(flows[0,:, :, 0], 0), 3 )))
    tf.image_summary('Flow-y Map', tf.to_float(tf.expand_dims(tf.expand_dims(flows[0,:, :, 1], 0), 3 )))
    tf.scalar_summary('Mean Flow', tf.reduce_mean(flows[0, :, :, :]))
    tf.scalar_summary('Max Flow', tf.reduce_max(flows[0, :, :, :]))
    tf.scalar_summary('Min Flow', tf.reduce_min(flows[0, :, :, :]))

    shape = inputs.get_shape()
    shape = [int(dim) for dim in shape]
    inputs = tf.reshape(inputs, [shape[0], -1, shape[3]])
    flows = tf.reshape(flows, [shape[0], -1, 2])
    floor_flows = tf.to_int32(tf.round(flows))
    weights_flows = flows-tf.floor(flows)

    pos_x = tf.range(shape[1])
    pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, shape[2]])
    pos_x = tf.reshape(pos_x, [-1])
    pos_y = tf.range(shape[2])
    pos_y = tf.tile(tf.expand_dims(pos_y, 0), [shape[1], 1])
    pos_y = tf.reshape(pos_y, [-1])
    batch = []
    for b in range(shape[0]):
        channel = []
        for c in range(shape[3]):
	  # predicted positions
	  pos1 = (floor_flows[b, :, 0] + pos_x)*shape[2] + (floor_flows[b, :, 1] + pos_y)
	  pos2 = (floor_flows[b, :, 0] + pos_x + 1)*shape[2] + (floor_flows[b, :, 1] + pos_y)
	  pos3 = (floor_flows[b, :, 0] + pos_x)*shape[2] + (floor_flows[b, :, 1] + pos_y + 1)
	  pos4 = (floor_flows[b, :, 0] + pos_x + 1)*shape[2] + (floor_flows[b, :, 1] + pos_y + 1)
	  
	  # get the corresponding pixels
	  pixel1 = tf.gather(inputs[b, :, c], pos1)
	  pixel2 = tf.gather(inputs[b, :, c], pos2)
	  pixel3 = tf.gather(inputs[b, :, c], pos3)
	  pixel4 = tf.gather(inputs[b, :, c], pos4)
	
	  # linear interpretation of these predicted pixels
	  # remove the linear interpretation because of OOM issue. 
	  xw = weights_flows[b, :, 0]
	  yw = weights_flows[b, :, 1]
          img = tf.mul(pixel1, (1-xw)*(1-yw)) + tf.mul(pixel2, xw*(1-yw)) + \
          	  tf.mul(pixel3, (1-xw)*yw) + tf.mul(pixel4, xw*yw) 
          img = tf.mul(pixel1, (1-yw)) + tf.mul(pixel3, yw)
          channel.append(pixel1)
        batch.append(tf.reshape(tf.transpose(tf.pack(channel)), [shape[1], shape[2], shape[3]]))
    preds = tf.pack(batch)


    loss = tf.contrib.losses.sum_of_squares(preds, outputs)
    slim.losses.add_loss(loss)
    tf.scalar_summary('Sum of squares loss', loss)
  return  loss, preds



