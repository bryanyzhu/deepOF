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

        # conv1_1 = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
        conv1_1 = slim.conv2d(tf.concat(3, [inputs, outputs]), 64, [3, 3], scope='conv1_1')
        pool1 = slim.max_pool2d(conv1_1, [2, 2], scope='pool1')
        conv2_1 = slim.conv2d(pool1, 128, [3, 3], scope='conv2_1')
        pool2 = slim.max_pool2d(conv2_1, [2, 2], scope='pool2')
        conv3_1 = slim.conv2d(pool2, 256, [3, 3], scope='conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 256, [3, 3], scope='conv3_2')
        pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')
        conv4_1 = slim.conv2d(pool3, 512, [3, 3], scope='conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 512, [3, 3], scope='conv4_2')
        pool4 = slim.max_pool2d(conv4_2, [2, 2], scope='pool4')
        conv5_1 = slim.conv2d(pool4, 512, [3, 3], scope='conv5_1')
        conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], scope='conv5_2')
        pool5 = slim.max_pool2d(conv5_2, [2, 2], scope='pool5')
        flatten5 = slim.flatten(pool5, scope='flatten5')
        fc6 = slim.fully_connected(flatten5, 4096, scope='fc6')
        dropout6 = slim.dropout(fc6, 0.5, scope='dropout6')
        fc7 = slim.fully_connected(dropout6, 4096, scope='fc7')
        dropout7 = slim.dropout(fc7, 0.5, scope='dropout7')
        
        channels = 40
        reshape_h = pool4.get_shape()[1].value/2
        reshape_w = pool4.get_shape()[2].value/2
     
        fc8 = slim.fully_connected(dropout7, channels*reshape_h*reshape_w, scope='d5')
        bn_pool1 = slim.batch_norm(pool1, scope='bn_pool1')
        bn_pool2 = slim.batch_norm(pool2, scope='bn_pool2')
        bn_pool3 = slim.batch_norm(pool3, scope='bn_pool3')
        bn_pool4 = slim.batch_norm(pool4, scope='bn_pool4')

        d1 = slim.conv2d(bn_pool1, channels, [3, 3], scope='d1')
        d2 = slim.conv2d(bn_pool2, channels, [3, 3], scope='d2')
        d3 = slim.conv2d(bn_pool3, channels, [3, 3], scope='d3')
        d4 = slim.conv2d(bn_pool4, channels, [3, 3], scope='d4')
        d5 = tf.reshape(fc8, [-1, reshape_h, reshape_w, channels])

        scale = 1
        d1 = slim.conv2d_transpose(d1, channels, [2*scale, 2*scale], stride=scale, scope='deconv_1')
        scale *= 2
        d2 = slim.conv2d_transpose(d2, channels, [2*scale, 2*scale], stride=scale, scope='deconv_2')
        scale *= 2
        d3 = slim.conv2d_transpose(d3, channels, [2*scale, 2*scale], stride=scale, scope='deconv_3')
        scale *= 2
        d4 = slim.conv2d_transpose(d4, channels, [2*scale, 2*scale], stride=scale, scope='deconv_4')
        scale *= 2
        d5 = slim.conv2d_transpose(d5, channels, [2*scale, 2*scale], stride=scale, scope='deconv_5')
        flows = tf.add_n([d1, d2, d3, d4, d5])

        scale = 2
        flows = slim.conv2d_transpose(flows, channels, [2*scale, 2*scale], stride=scale, scope='deconv_final')
        flows = slim.conv2d(flows, 2, [3, 3], activation_fn=None, scope='final_conv')


        shape = inputs.get_shape()
        shape = [int(dim) for dim in shape]
        inputs_flat = tf.reshape(inputs, [shape[0], -1, shape[3]])
        outputs_flat = tf.reshape(outputs, [shape[0], -1, shape[3]])

        # TODO: Median filtering the flow 
        flows = tf.reshape(flows, [shape[0], -1, 2])
        floor_flows = tf.to_int32(tf.floor(flows))
        weights_flows = flows - tf.floor(flows)

        pos_x = tf.range(shape[1])
        pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, shape[2]])
        pos_x = tf.reshape(pos_x, [-1])
        pos_y = tf.range(shape[2])
        pos_y = tf.tile(tf.expand_dims(pos_y, 0), [shape[1], 1])
        pos_y = tf.reshape(pos_y, [-1])


        batch, batch_1 = [], []
        for b in range(shape[0]):
            channel, channel_1 = [], []
            for c in range(shape[3]):
                # predicted positions
                pos1 = (pos_x + floor_flows[b, :, 0])*shape[2] + (pos_y + floor_flows[b, :, 1] )
                pos2 = (pos_x + floor_flows[b, :, 0] + 1)*shape[2] + (pos_y + floor_flows[b, :, 1] )
                pos3 = (pos_x + floor_flows[b, :, 0])*shape[2] + (pos_y + floor_flows[b, :, 1] + 1)
                pos4 = (pos_x + floor_flows[b, :, 0] + 1)*shape[2] + (pos_y + floor_flows[b, :, 1] + 1)
                pos5 = (pos_x - floor_flows[b, :, 0])*shape[2] + (pos_y - floor_flows[b, :, 1] )
                pos6 = (pos_x - floor_flows[b, :, 0] - 1)*shape[2] + (pos_y - floor_flows[b, :, 1] )
                pos7 = (pos_x - floor_flows[b, :, 0])*shape[2] + (pos_y - floor_flows[b, :, 1] - 1)
                pos8 = (pos_x - floor_flows[b, :, 0] - 1)*shape[2] + (pos_y - floor_flows[b, :, 1] - 1)

                zero = tf.zeros([], dtype='int32')
                pos1 = tf.clip_by_value(pos1, zero, shape[1]*shape[2])
                pos2 = tf.clip_by_value(pos2, zero, shape[1]*shape[2])
                pos3 = tf.clip_by_value(pos3, zero, shape[1]*shape[2])
                pos4 = tf.clip_by_value(pos4, zero, shape[1]*shape[2])
                pos5 = tf.clip_by_value(pos5, zero, shape[1]*shape[2])
                pos6 = tf.clip_by_value(pos6, zero, shape[1]*shape[2])
                pos7 = tf.clip_by_value(pos7, zero, shape[1]*shape[2])
                pos8 = tf.clip_by_value(pos8, zero, shape[1]*shape[2])

                # get the corresponding pixels
                pixel1 = tf.gather(inputs_flat[b, :, c], pos1)
                pixel2 = tf.gather(inputs_flat[b, :, c], pos2)
                pixel3 = tf.gather(inputs_flat[b, :, c], pos3)
                pixel4 = tf.gather(inputs_flat[b, :, c], pos4)
                pixel5 = tf.gather(outputs_flat[b, :, c], pos5)
                pixel6 = tf.gather(outputs_flat[b, :, c], pos6)
                pixel7 = tf.gather(outputs_flat[b, :, c], pos7)
                pixel8 = tf.gather(outputs_flat[b, :, c], pos8)

                # linear interpretation of these predicted pixels
                xw = weights_flows[b, :, 0]
                yw = weights_flows[b, :, 1]
                img = tf.mul(pixel1, (1-xw)*(1-yw)) + tf.mul(pixel2, xw*(1-yw)) + \
                          tf.mul(pixel3, (1-xw)*yw) + tf.mul(pixel4, xw*yw)
                img_1 = tf.mul(pixel5, (1-xw)*(1-yw)) + tf.mul(pixel6, xw*(1-yw)) + \
                          tf.mul(pixel7, (1-xw)*yw) + tf.mul(pixel8, xw*yw)
                channel.append(img)
                channel_1.append(img_1)
            batch.append(tf.transpose(tf.pack(channel)))
            batch_1.append(tf.transpose(tf.pack(channel_1)))
        preds = tf.pack(batch)
        reconstructs = tf.pack(batch_1)

        loss_predict = tf.contrib.losses.sum_of_squares(preds, outputs_flat)
        loss_reconstruct = tf.contrib.losses.sum_of_squares(reconstructs, inputs_flat)
        # slim.losses.add_loss(tf.minimum(loss_predict, loss_reconstruct))

        # Charbonnier penalty function
        epsilon = 0.001
        alpha_c = 0.45
        loss_min = tf.minimum(loss_predict, loss_reconstruct)
        Charbonnier = tf.pow(loss_min + tf.square(tf.constant(epsilon, name='epsilon')), tf.constant(alpha_c, name='alpha_c'))

        # Smoothness loss
        alpha_s = 0.45
        flow_vis = tf.reshape(flows,[shape[0], shape[1], shape[2], 2])
        flowx = flow_vis[:,:,:,0]
        flowy = flow_vis[:,:,:,1]
        a = tf.reduce_mean(tf.pow(tf.square(tf.sub(flowx[:, :-1, :], flowx[:, 1:, :])) + tf.square(tf.constant(epsilon)), tf.constant(alpha_s)))          # u(i,j) - u(i+1,j)
        b = tf.reduce_mean(tf.pow(tf.square(tf.sub(flowx[:, :, :-1], flowx[:, :, 1:])) + tf.square(tf.constant(epsilon)), tf.constant(alpha_s)))          # u(i,j) - u(i,j+1)
        c = tf.reduce_mean(tf.pow(tf.square(tf.sub(flowy[:, :-1, :], flowy[:, 1:, :])) + tf.square(tf.constant(epsilon)), tf.constant(alpha_s)))          # v(i,j) - v(i+1,j)
        d = tf.reduce_mean(tf.pow(tf.square(tf.sub(flowy[:, :, :-1], flowy[:, :, 1:])) + tf.square(tf.constant(epsilon)), tf.constant(alpha_s)))          # v(i,j) - v(i,j+1)
        loss_smooth = a + b + c + d

        slim.losses.add_loss(Charbonnier)
        slim.losses.add_loss(loss_smooth)

    # return loss_predict, loss_reconstruct, flow_vis, tf.reshape(preds, [shape[0], shape[1], shape[2], shape[3]])
    return Charbonnier, loss_smooth, flow_vis, tf.reshape(preds, [shape[0], shape[1], shape[2], shape[3]])







