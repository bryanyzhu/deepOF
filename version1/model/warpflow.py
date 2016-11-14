import tensorflow as tf
import numpy as np

def loss_interp(flows, inputs, outputs, epsilon, alpha_c, alpha_s, lambda_smooth, flow_scale, deltaWeights):

    num_batch = inputs.get_shape()[0].value
    height = inputs.get_shape()[1].value
    width = inputs.get_shape()[2].value
    channels = inputs.get_shape()[3].value
    flow_channels = channels/3*2

    needMask = deltaWeights["needMask"]
    needImageGradients = deltaWeights["needImageGradients"]
    # Create border mask for image
    border_ratio = 0.1
    shortestDim = height
    borderWidth = int(np.ceil(shortestDim * border_ratio))
    smallerMask = tf.ones([height-2*borderWidth, width-2*borderWidth])
    borderMask = tf.pad(smallerMask, [[borderWidth,borderWidth], [borderWidth,borderWidth]], "CONSTANT")
    borderMask = tf.tile(tf.expand_dims(borderMask, 0), [num_batch, 1, 1])
    borderMaskImg = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, channels])
    borderMaskFlow = tf.tile(tf.expand_dims(borderMask, 3), [1, 1, 1, flow_channels])

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

    scaled_flows = tf.mul(flows, flow_scale)
    flows_flat = tf.reshape(scaled_flows, [num_batch, -1, flow_channels])
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
    
    # Calculating image gradients
    if needImageGradients:
        rgb_images_list = []
        for b_idx in xrange(num_batch):
            image_idx = inputs[b_idx,:,:,:]
            max_value = tf.reduce_max(image_idx)
            min_value = tf.reduce_min(image_idx)
            intensity_range = max_value - min_value
            image_idx = tf.truediv(tf.scalar_mul(255.0, tf.sub(image_idx, min_value)), intensity_range)
            image_idx_clip = tf.clip_by_value(tf.to_int32(image_idx), zero, 255)
            rgb_images_list.append(image_idx_clip)

        rgb_images = tf.pack(rgb_images_list, axis=0)
        # result.append(rgb_images)
        inputs_gray = tf.to_float(tf.image.rgb_to_grayscale(rgb_images))
        # result.append(inputs_gray)
        img_gradients_horizontal = tf.nn.depthwise_conv2d(inputs_gray, deltaWeights["sobel_x_filter"], [1,1,1,1], padding="SAME")
        img_gradients_vertical = tf.nn.depthwise_conv2d(inputs_gray, deltaWeights["sobel_y_filter"], [1,1,1,1], padding="SAME")
        img_gradients_horizontal = tf.div(img_gradients_horizontal, tf.reduce_max(tf.abs(img_gradients_horizontal)))
        img_gradients_vertical = tf.div(img_gradients_vertical, tf.reduce_max(tf.abs(img_gradients_vertical)))
        # gradientsMag = tf.sqrt(tf.pow(img_gradients_horizontal, 2) + tf.pow(img_gradients_vertical, 2))
        
        eta = 1.0
        gradientsMaskFlow_x = tf.scalar_mul(eta, tf.sub(1.0, tf.abs(img_gradients_horizontal)))
        gradientsMaskFlow_y = tf.scalar_mul(eta, tf.sub(1.0, tf.abs(img_gradients_vertical)))
        gradientsMaskFlow = tf.concat(3, [gradientsMaskFlow_x, gradientsMaskFlow_y])

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
    horizontal_gradients = tf.nn.depthwise_conv2d(flows, deltaWeights["flow_width_filter"], [1,1,1,1], padding="SAME") 
    vertical_gradients   = tf.nn.depthwise_conv2d(flows, deltaWeights["flow_height_filter"], [1,1,1,1], padding="SAME") 
    U_delta = tf.pack([horizontal_gradients[:,:,:,0], vertical_gradients[:,:,:,0]], axis=3)
    V_delta = tf.pack([horizontal_gradients[:,:,:,1], vertical_gradients[:,:,:,1]], axis=3)
    
    U_loss = 0.0
    V_loss = 0.0
    numValidFlows = numValidPixels/3*2
    if needMask:
        U_delta_clean = tf.mul(U_delta, smoothnessMask)
        V_delta_clean = tf.mul(V_delta, smoothnessMask)

        eleWiseULoss = tf.pow(tf.square(U_delta_clean) + tf.square(epsilon), alpha_s)
        #result.append(eleWiseULoss)
        
        if needImageGradients:
            eleWiseULoss = tf.mul(gradientsMaskFlow, eleWiseULoss)
            #result.append(eleWiseULoss)
        eleWiseULoss = tf.mul(borderMaskFlow, eleWiseULoss)
        U_loss = tf.reduce_sum(eleWiseULoss) / numValidFlows

        eleWiseVLoss = tf.pow(tf.square(V_delta_clean) + tf.square(epsilon), alpha_s)
        if needImageGradients:
            eleWiseVLoss = tf.mul(gradientsMaskFlow, eleWiseVLoss)
        eleWiseVLoss = tf.mul(borderMaskFlow, eleWiseVLoss)
        V_loss = tf.reduce_sum(eleWiseVLoss) / numValidFlows
    else:
        U_loss = tf.reduce_mean(tf.pow(tf.square(U_delta)  + tf.square(epsilon), alpha_s)) 
        V_loss = tf.reduce_mean(tf.pow(tf.square(V_delta)  + tf.square(epsilon), alpha_s))
    loss_smooth = U_loss + V_loss

    total_loss = Charbonnier_reconstruct + lambda_smooth * loss_smooth
    # Define a loss structure
    lossDict = {}
    lossDict["total"] = total_loss
    lossDict["Charbonnier_reconstruct"] = Charbonnier_reconstruct
    lossDict["U_loss"] = U_loss
    lossDict["V_loss"] = V_loss
    #lossDict["result"] = result

    return lossDict, tf.reshape(reconstructs, [num_batch, height, width, 3])