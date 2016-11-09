import os, sys
import numpy as np
import cv2

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            #print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (h, w, 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def norm(arr, axis=-1):
    return np.sqrt(np.sum(arr**2, axis=axis))

def div_nonz(a,b):
    anz = a[b != 0]
    bnz = b[b != 0]
    result = np.zeros_like(a)
    result[b != 0] = anz / bnz
    return result

def flow_ee(f1, f2, mask=None):
	ee_tot = np.sqrt((f1[:,:,:,0] - f2[:,:,:,0])**2 + (f1[:,:,:,1] - f2[:,:,:,1])**2) 
	aee = np.mean(ee_tot, axis=None) 
	# return ee_tot, aee
	return aee

def flow_ae(f1, f2, mask=None):
	u = f1[:,:,:,0]
	u_GT = f2[:,:,:,0]
	v = f1[:,:,:,1]
	v_GT = f2[:,:,:,1]
	numerator = 1 + u * u_GT + v * v_GT
	denominator = np.sqrt(1 + u**2 + v**2) * np.sqrt(1 + u_GT**2 + v_GT**2)
	ae_tot = np.arccos(np.clip(numerator / denominator, -1, 1))
	aae = np.mean(ae_tot, axis=None) 
	# return ae_tot, aae
	return aae

def geoAugmentation(source, target):
    """
    Includes random flip, translation, scale, rotation
    """
    img0List = []
    img1List = []
    batch_size = source.shape[0]
    height = source.shape[1]
    width = source.shape[2]

    for batch_idx in xrange(batch_size):
        img0 = source[batch_idx,:,:,:]
        img1 = target[batch_idx,:,:,:]

        # translation
        translate_x = np.random.uniform(low=-0.2, high=0.2, size=1)
        translate_y = np.random.uniform(low=-0.2, high=0.2, size=1)
        x_move = int(translate_x * width)
        y_move = int(translate_y * height)

        translation_matrix = np.float32([ [1,0,x_move], [0,1,y_move] ])
        img0_translation = cv2.warpAffine(img0, translation_matrix, (width, height))
        img1_translation = cv2.warpAffine(img1, translation_matrix, (width, height))

        # img0List.append(np.expand_dims(img0_translation, 0))
        # img1List.append(np.expand_dims(img1_translation, 0))
        # print img0_translation.shape
        # print img1_translation.shape

        # rotation
        rotation_ratio = np.random.uniform(low=-17, high=17, size=1)
        center = (width / 2, height / 2)
        M = cv2.getRotationMatrix2D(center, rotation_ratio, 1.0)
        rotated_img0 = cv2.warpAffine(img0_translation, M, (width, height))
        rotated_img1 = cv2.warpAffine(img1_translation, M, (width, height))
        # img0List.append(np.expand_dims(rotated_img0, 0))
        # img1List.append(np.expand_dims(rotated_img1, 0))
        # print rotated_img0.shape
        # print rotated_img1.shape 

        # scale
        scale_ratio = np.random.uniform(low=0.9, high=2.0, size=1)
        scaled_width = int(width * scale_ratio)
        scaled_height = int(height * scale_ratio)
        img0_scale = 0
        img1_scale = 0

        left_move, right_move, up_move, down_move = 0, 0, 0, 0
        if scale_ratio > 1.0:
            img0_scale = cv2.resize(rotated_img0, (scaled_width, scaled_height))
            img1_scale = cv2.resize(rotated_img1, (scaled_width, scaled_height))
            if (scaled_width - width) % 2 == 0:
                left_move = (scaled_width - width) / 2 
                right_move = left_move
            else:
                left_move = (scaled_width - width - 1) / 2 
                right_move = left_move + 1
            if (scaled_height - height) % 2 == 0:
                up_move = (scaled_height - height) / 2 
                down_move = up_move
            else:
                up_move = (scaled_height - height - 1) / 2 
                down_move = up_move + 1

            # print up_move, down_move, left_move, right_move
            cond1 = (up_move == 0 and down_move == 0)
            cond2 = (left_move == 0 and right_move == 0)
            if  cond1 and cond2:
                img0_scale = img0_scale[:, :, :]
                img1_scale = img1_scale[:, :, :]
            elif cond1 and not cond2:
                img0_scale = img0_scale[:, left_move:-right_move, :]
                img1_scale = img1_scale[:, left_move:-right_move, :]
            elif not cond1 and cond2:
                img0_scale = img0_scale[up_move:-down_move, :, :]
                img1_scale = img1_scale[up_move:-down_move, :, :]
            else:
                img0_scale = img0_scale[up_move:-down_move, left_move:-right_move, :]
                img1_scale = img1_scale[up_move:-down_move, left_move:-right_move, :]
            # print "cropping"
            # print img0_scale.shape
            # print img1_scale.shape
            # img0List.append(np.expand_dims(img0_scale, 0))
            # img1List.append(np.expand_dims(img1_scale, 0))

        elif scale_ratio < 1.0:
            img0_scale = cv2.resize(rotated_img0, (scaled_width, scaled_height))
            img1_scale = cv2.resize(rotated_img1, (scaled_width, scaled_height))
            if (width - scaled_width) % 2 == 0:
                left_move = (width - scaled_width) / 2 
                right_move = left_move
            else:
                left_move = (width - scaled_width - 1) / 2 
                right_move = left_move + 1
            if (height - scaled_height) % 2 == 0:
                up_move = (height - scaled_height) / 2 
                down_move = up_move
            else:
                up_move = (height - scaled_height - 1) / 2 
                down_move = up_move + 1

            img0_scale = cv2.copyMakeBorder(img0_scale,up_move,down_move,left_move,right_move,cv2.BORDER_CONSTANT,value=0)       # top, bottom, left, right
            img1_scale = cv2.copyMakeBorder(img1_scale,up_move,down_move,left_move,right_move,cv2.BORDER_CONSTANT,value=0)
            # print "padding"
            # print img0_scale.shape
            # print img1_scale.shape 
            # img0List.append(np.expand_dims(img0_scale, 0))
            # img1List.append(np.expand_dims(img1_scale, 0))

        else:
            img0_scale = rotated_img0
            img1_scale = rotated_img1
            # img0List.append(np.expand_dims(img0, 0))
            # img1List.append(np.expand_dims(img1, 0))
        
        # random flip
        flip_prob = np.random.uniform(low=0.0, high=1.0, size=1)
        img0_flip = 0
        img1_flip = 0
        if flip_prob >= 0.5:
            img0_flip = np.fliplr(img0_scale)
            img1_flip = np.fliplr(img1_scale)
            # img0List.append(np.expand_dims(np.fliplr(img0), 0))
            # img1List.append(np.expand_dims(np.fliplr(img1), 0))
            # print np.fliplr(img0).shape
        else:
            img0_flip = img0_scale
            img1_flip = img1_scale
            # img0List.append(np.expand_dims(img0, 0))
            # img1List.append(np.expand_dims(img1, 0))
            # print img0.shape

        img0List.append(np.expand_dims(img0_flip, 0))
        img1List.append(np.expand_dims(img1_flip, 0))

    return np.concatenate(img0List, axis=0), np.concatenate(img1List, axis=0)

def photoAugmentation(source, target, mean):
    """
    Includes additive gaussian noise, changes in brightness, contrast, gamma and color
    """
    img0List = []
    img1List = []
    batch_size = source.shape[0]
    height = source.shape[1]
    width = source.shape[2]

    for batch_idx in xrange(batch_size):
        img0 = source[batch_idx,:,:,:]
        img1 = target[batch_idx,:,:,:]
        # print "origin"
        # print np.max(img0), np.min(img1)

        # contrast and brightness change
        contrast = np.random.uniform(low=-0.8, high=0.4, size=1)
        bright_sigma = np.random.uniform(low=0, high=0.2, size=1)
        brightnessImage = np.random.normal(0, bright_sigma, ([height,width,3]))
        img0_contrast = contrast * img0 + brightnessImage
        img1_contrast = contrast * img1 + brightnessImage
        # print "contrast"
        # print np.max(img0_contrast), np.min(img1_contrast)
        
        # color change
        color_changes = np.random.uniform(low=0.5, high=2, size=3)
        color_change_B = color_changes[0]
        color_change_G = color_changes[1]
        color_change_R = color_changes[2]
        img0_color_B = img0_contrast[:,:,0] * color_change_B
        img0_color_G = img0_contrast[:,:,1] * color_change_G
        img0_color_R = img0_contrast[:,:,2] * color_change_R
        img0_color = np.stack([img0_color_B, img0_color_G, img0_color_R], axis=-1)
        img1_color_B = img1_contrast[:,:,0] * color_change_B
        img1_color_G = img1_contrast[:,:,1] * color_change_G
        img1_color_R = img1_contrast[:,:,2] * color_change_R
        img1_color = np.stack([img1_color_B, img1_color_G, img1_color_R], axis=-1)
        # print "color"
        # print np.max(img0_color), np.min(img0_color)
        # clip value
        img0_color = np.clip(img0_color, 0, 1)
        img1_color = np.clip(img1_color, 0, 1)

        # gamma
        gamma = np.random.uniform(low=0.7, high=1.5, size=1)
        gamma_inv = 1 / gamma
        img0_gamma = np.power(img0_color, gamma_inv)
        img1_gamma = np.power(img1_color, gamma_inv)
        # print "gamma"
        # print np.max(img0_gamma), np.min(img0_gamma)

        # additive gaussian noise
        sigma = np.random.uniform(low=0, high=0.04, size=1)
        noiseImage = np.random.normal(0, sigma, ([height,width,3]))
        img0_noise = img0_gamma + noiseImage
        img1_noise = img1_gamma + noiseImage
        # print "noise"
        # print np.max(img0_noise), np.min(img0_noise)

        # subtract mean
        img0_noise[:,:,0] = img0_noise[:,:,0] - mean[0]/255.0
        img0_noise[:,:,1] = img0_noise[:,:,1] - mean[1]/255.0
        img0_noise[:,:,2] = img0_noise[:,:,2] - mean[2]/255.0
        img1_noise[:,:,0] = img1_noise[:,:,0] - mean[0]/255.0
        img1_noise[:,:,1] = img1_noise[:,:,1] - mean[1]/255.0
        img1_noise[:,:,2] = img1_noise[:,:,2] - mean[2]/255.0
        # print "mean"
        # print np.max(img0_noise), np.min(img0_noise)

        img0List.append(np.expand_dims(img0_noise, 0))
        img1List.append(np.expand_dims(img1_noise, 0))
        # print "one sample done"

    return np.concatenate(img0List, axis=0), np.concatenate(img1List, axis=0)

def flowToColor(flow):
	UNKNOWN_FLOW_THRESH = 1e9;
	UNKNOWN_FLOW = 1e10;            

	height, width, nBands = flow.shape

	if nBands != 2:
	    print('flowToColor: flow image must have two bands')    

	u = flow[:,:,0]
	v = flow[:,:,1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999
	maxrad = -1

	# fix unknown flow
	idxUnknown = (abs(u)> UNKNOWN_FLOW_THRESH) | (abs(v)> UNKNOWN_FLOW_THRESH)
	u[idxUnknown] = 0
	v[idxUnknown] = 0

	maxu = np.maximum(maxu, np.amax(u, axis=None))
	minu = np.minimum(minu, np.amin(u, axis=None))

	maxv = np.maximum(maxv, np.amax(v, axis=None))
	minv = np.minimum(minv, np.amin(v, axis=None))

	rad = np.sqrt(u**2 + v**2)
	maxrad = np.maximum(maxrad, np.amax(rad, axis=None))

	# fprintf('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n', maxrad, minu, maxu, minv, maxv);

	# if isempty(varargin) == 0:
	#     maxFlow = varargin{1};
	#     if maxFlow > 0
	#         maxrad = maxFlow;
	#     end;       
	# end;
	eps = 2.22e-16
	u = u / (maxrad + eps)
	v = v / (maxrad + eps)

	# compute color
	img = computeColor(u, v)
	return img 
	    
	# % unknown flow
	# IDX = repmat(idxUnknown, [1 1 3]);
	# img(IDX) = 0;

def computeColor(u,v,logscale=False,scaledown=1,output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u**2 + v**2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown    
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    rot = np.arctan2(-v, -u) / np.pi

    fk = (rot+1)/2 * (ncols-1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)       # 0, 1, 2, ..., ncols

    k1 = k0+1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape+(ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1-f)*col0 + f*col1
       
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        # out of range    
        col[~idx] *= 0.75
        img[:,:,i] = np.floor(255*col).astype(np.uint8)
    
    return img.astype(np.uint8)
    
def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros((ncols,3))
    
    col = 0
    # RY
    colorwheel[0:RY,0] = 1
    colorwheel[0:RY,1] = np.arange(0,1,1./RY)
    col += RY
    
    # YG
    colorwheel[col:col+YG,0] = np.arange(1,0,-1./YG)
    colorwheel[col:col+YG,1] = 1
    col += YG
    
    # GC
    colorwheel[col:col+GC,1] = 1
    colorwheel[col:col+GC,2] = np.arange(0,1,1./GC)
    col += GC
    
    # CB
    colorwheel[col:col+CB,1] = np.arange(1,0,-1./CB)
    colorwheel[col:col+CB,2] = 1
    col += CB
    
    # BM
    colorwheel[col:col+BM,2] = 1
    colorwheel[col:col+BM,0] = np.arange(0,1,1./BM)
    col += BM
    
    # MR
    colorwheel[col:col+MR,2] = np.arange(1,0,-1./MR)
    colorwheel[col:col+MR,0] = 1

    return colorwheel    
  