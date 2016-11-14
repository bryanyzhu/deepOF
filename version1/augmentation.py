import tensorflow as tf

def geoAugmentation(source, target):
    """
    Includes translation, scale, random flip
    """
    num_batch = source.get_shape()[0].value
    height = source.get_shape()[1].value
    width = source.get_shape()[2].value

    geo_source_list = []
    geo_target_list = []
    for batch_idx in xrange(num_batch):
        img0 = source[batch_idx,:,:,:]
        img1 = target[batch_idx,:,:,:]

        # Translation (implemented as cropping)
        translate_x = tf.random_uniform([], minval=-0.2, maxval=0.2)
        translate_y = tf.random_uniform([], minval=-0.2, maxval=0.2)
        x_move = tf.cast(tf.abs(translate_x) * width,  dtype=tf.int32)
        y_move = tf.cast(tf.abs(translate_y) * height, dtype=tf.int32)
        hor_move = tf.cond(tf.greater(translate_x, 0.0), lambda: x_move-x_move, lambda: x_move)
        ver_move = tf.cond(tf.greater(translate_y, 0.0), lambda: y_move-y_move, lambda: y_move)
        img0_translated = tf.image.crop_to_bounding_box(img0, ver_move, hor_move, height-y_move, width-x_move)
        img1_translated = tf.image.crop_to_bounding_box(img1, ver_move, hor_move, height-y_move, width-x_move)
        
        # No random rotation because no existing tensorflow implementations

        # Scale
        scale_ratio = tf.random_uniform([], minval=0.9, maxval=2.0)
        scaled_width = tf.cast(scale_ratio * width,  dtype=tf.int32)
        scaled_height = tf.cast(scale_ratio * height,  dtype=tf.int32)
        img0_scaled = tf.image.resize_images(img0_translated, [scaled_height, scaled_width])
        img1_scaled = tf.image.resize_images(img1_translated, [scaled_height, scaled_width])
        img0_crop_pad = tf.image.resize_image_with_crop_or_pad(img0_scaled, height, width)
        img1_crop_pad = tf.image.resize_image_with_crop_or_pad(img1_scaled, height, width)

        # Random flip left and right
        flip_ratio = tf.random_uniform([], minval=0.0, maxval=1.0)
        img0_flipped = tf.cond(tf.greater(flip_ratio, 0.5), lambda: tf.image.flip_left_right(img0_crop_pad), lambda: img0_crop_pad)
        img1_flipped = tf.cond(tf.greater(flip_ratio, 0.5), lambda: tf.image.flip_left_right(img1_crop_pad), lambda: img1_crop_pad)

        geo_source_list.append(img0_flipped)
        geo_target_list.append(img1_flipped)

    return tf.pack(geo_source_list, axis=0), tf.pack(geo_target_list, axis=0)

def photoAugmentation(source, target, mean):
    """
    Includes contrast and brightness, color channel and gamma change, adding additive gaussian noise
    """
    num_batch = source.get_shape()[0].value
    height = source.get_shape()[1].value
    width = source.get_shape()[2].value

    photo_source_list = []
    photo_target_list = []
    for batch_idx in xrange(num_batch):
        img0 = source[batch_idx,:,:,:]
        img1 = target[batch_idx,:,:,:]

        # Contrast and brightness change
        contrast = tf.random_uniform([], minval=-0.3, maxval=0.3)
        contrast = contrast + 1.0
        bright_sigma = 0.2    # tf.random_uniform([], minval=0.0, maxval=0.2)
        brightnessImage = tf.random_normal([height,width,3], mean=0.0, stddev=bright_sigma, dtype=tf.float32)
        img0_contrast = tf.add(tf.scalar_mul(contrast, img0), brightnessImage)
        img1_contrast = tf.add(tf.scalar_mul(contrast, img1), brightnessImage)
        
        # Color change, may be bad for unsupervised learning
        color_change_B = tf.random_uniform([], minval=0.9, maxval=1.1)
        color_change_G = tf.random_uniform([], minval=0.9, maxval=1.1)
        color_change_R = tf.random_uniform([], minval=0.9, maxval=1.1)
        img0_color_B = tf.scalar_mul(color_change_B, img0_contrast[:,:,0])
        img0_color_G = tf.scalar_mul(color_change_G, img0_contrast[:,:,1])
        img0_color_R = tf.scalar_mul(color_change_R, img0_contrast[:,:,2])
        img0_color = tf.pack([img0_color_B, img0_color_G, img0_color_R], axis=2)
        img1_color_B = tf.scalar_mul(color_change_B, img1_contrast[:,:,0])
        img1_color_G = tf.scalar_mul(color_change_G, img1_contrast[:,:,1])
        img1_color_R = tf.scalar_mul(color_change_R, img1_contrast[:,:,2])
        img1_color = tf.pack([img1_color_B, img1_color_G, img1_color_R], axis=2)

        img0_color = tf.clip_by_value(img0_color, 0.0, 1.0)
        img1_color = tf.clip_by_value(img1_color, 0.0, 1.0)

        # Gamma
        gamma = tf.random_uniform([], minval=0.7, maxval=1.5)
        gamma_inv = tf.inv(gamma)
        img0_gamma = tf.pow(img0_color, gamma_inv)
        img1_gamma = tf.pow(img1_color, gamma_inv)
        
        # Additive gaussian noise
        sigma = tf.random_uniform([], minval=0.0, maxval=0.04)
        noiseImage = tf.random_normal([height,width,3], mean=0.0, stddev=sigma, dtype=tf.float32)
        img0_noise = tf.add(img0_gamma, noiseImage)
        img1_noise = tf.add(img1_gamma, noiseImage)

        # Subtract mean
        img0_mean = tf.sub(img0_noise, tf.truediv(mean, 255.0))
        img1_mean = tf.sub(img1_noise, tf.truediv(mean, 255.0))

        photo_source_list.append(img0_mean)
        photo_target_list.append(img1_mean)

    return tf.pack(photo_source_list, axis=0), tf.pack(photo_target_list, axis=0)