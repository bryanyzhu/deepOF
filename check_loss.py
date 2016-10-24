import os,sys
import numpy as np
import matplotlib.pyplot as pp
import cv2

pr1 = np.load("/home/yzhu25/Documents/deepOF/flows_1.npy")
pr2 = np.load("/home/yzhu25/Documents/deepOF/flows_2.npy")
pr3 = np.load("/home/yzhu25/Documents/deepOF/flows_3.npy")
pr4 = np.load("/home/yzhu25/Documents/deepOF/flows_4.npy")
pr5 = np.load("/home/yzhu25/Documents/deepOF/flows_5.npy")
pr6 = np.load("/home/yzhu25/Documents/deepOF/flows_6.npy")
source = np.load("/home/yzhu25/Documents/deepOF/source.pkl.npy")
target = np.load("/home/yzhu25/Documents/deepOF/target.pkl.npy")
flow_gt = np.load("/home/yzhu25/Documents/deepOF/flow_gt.pkl.npy")


#cv2.imshow('dst_rt', source[0,:,:,:])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

batch_num = 0

img1 = source[batch_num,:,:,:]
img2 = target[batch_num,:,:,:]
#print img1.shape
print np.mean(np.power(np.square(img1-img2) + (0.001)**2, 0.25))



mean = np.array([97.533268117955444, 99.238235788550085, 97.055973199626948])
inputs = img1 - mean
outputs = img2 - mean
# Scaling to 0 ~ 1 or -0.4 ~ 0.6?
inputs = np.true_divide(inputs, 255.0)
outputs = np.true_divide(outputs, 255.0)



#print np.mean(np.power(np.square((pr6_input-pr6_output))*255.0, 0.25))

pr6 = pr6[batch_num,:,:,:]*0.3125
pr6_input = cv2.resize(inputs, (pr6.shape[1], pr6.shape[0]))
#print pr6_input.shape
pr6_output = cv2.resize(outputs, (pr6.shape[1], pr6.shape[0]))
#print pr6.shape
#print np.max(pr6)
#print np.min(pr6)
#print np.mean(np.abs(pr6))

shape_dim = pr6_input.shape
height = shape_dim[0]
width = shape_dim[1]
channels = shape_dim[2]

inputs_flat = np.reshape(pr6_input, [-1, channels])
outputs_flat = np.reshape(pr6_output, [-1, channels])

print np.mean(np.power(np.square((inputs_flat-outputs_flat))*255.0, 0.25))


flows = pr6
flows = np.reshape(flows, [-1, 2])
#print flows.shape
floor_flows = np.floor(flows)
weights_flows = flows - floor_flows

#print np.max(floor_flows)
#print np.min(floor_flows)
#print np.mean(np.abs(floor_flows))

xx = np.linspace(0, height-1, height)
yy = np.linspace(0, width-1, width)
pos_y, pos_x = np.meshgrid(yy, xx)
pos_x = np.reshape(pos_x, [-1,])
pos_y = np.reshape(pos_y, [-1,])
#print pos_x.shape
#print pos_y.shape
#print pos_x
#print pos_y

channel = []
x = floor_flows[:, 0]
y = floor_flows[:, 1]
#print x.shape
#print y.shape
xw = weights_flows[:, 0]
yw = weights_flows[:, 1]

reconstructs = []
for c in xrange(channels):
	x0 = pos_y + x
	x1 = x0 + 1
	y0 = pos_x + y
	y1 = y0 + 1

	#print x0.shape
	#print x1
	#print y0
	#print y1

	idx_a = y0 * width + x0
	idx_b = y1 * width + x0
	idx_c = y0 * width + x1
	idx_d = y1 * width + x1

	idx_a = np.clip(idx_a, 0, height*width-1)
	idx_b = np.clip(idx_b, 0, height*width-1)
	idx_c = np.clip(idx_c, 0, height*width-1)
	idx_d = np.clip(idx_d, 0, height*width-1)

	wa = (1-xw) * (1-yw)
	wb = (1-xw) * yw
	wc = xw * (1-yw)
	wd = xw * yw

	#print idx_a
	#print idx_b
	#print idx_c
	#print idx_d
	#print outputs_flat.shape

	Ia = outputs_flat[:,c][idx_a.astype(int)]
	Ib = outputs_flat[:,c][idx_b.astype(int)]
	Ic = outputs_flat[:,c][idx_c.astype(int)]
	Id = outputs_flat[:,c][idx_d.astype(int)]
	# print wa, wb, wc, wd

	img = Ia*wa + Ib*wb + Ic*wc + Id*wd

	#print outputs_flat[:,c]
	#print Ia
	#print Ia.shape
	reconstructs.append(img)

recons = np.transpose(np.asarray(reconstructs))
#print recons.shape

print np.mean(np.power(np.square((inputs_flat-recons))*255.0, 0.25))



# batch = []
# for b in range(num_batch):
#     channel = []
#     x = floor_flows[b, :, 0]
#     y = floor_flows[b, :, 1]
#     xw = weights_flows[b, :, 0]
#     yw = weights_flows[b, :, 1]

#     for c in range(channels):

#         x0 = pos_y + x
#         x1 = x0 + 1
#         y0 = pos_x + y
#         y1 = y0 + 1

#         cond_a = tf.logical_and((x0 >= 0), (x0 < width-1))
#         cond_b = tf.logical_and((x1 >= 0), (x1 < width-1))
#         cond_c = tf.logical_and((y0 >= 0), (y0 < height-1))
#         cond_d = tf.logical_and((y1 >= 0), (y1 < height-1))
#         # x0 = tf.clip_by_value(x0, zero, width-1)
#         # x1 = tf.clip_by_value(x1, zero, width-1)
#         # y0 = tf.clip_by_value(y0, zero, height-1)
#         # y1 = tf.clip_by_value(y1, zero, height-1)

#         idx_a = y0 * width + x0
#         idx_b = y1 * width + x0
#         idx_c = y0 * width + x1
#         idx_d = y1 * width + x1

#         Ia = tf.gather(outputs_flat[b, :, c], idx_a)
#         Ib = tf.gather(outputs_flat[b, :, c], idx_b)
#         Ic = tf.gather(outputs_flat[b, :, c], idx_c)
#         Id = tf.gather(outputs_flat[b, :, c], idx_d)

#         wa = (1-xw) * (1-yw)
#         wb = (1-xw) * yw
#         wc = xw * (1-yw)
#         wd = xw * yw

#         # linear interpretation of these predicted pixels
#         img = tf.mul(Ia, tf.select(tf.logical_and(cond_a, cond_c), wa, zero_f)) + \
#               tf.mul(Ib, tf.select(tf.logical_and(cond_a, cond_d), wb, zero_f)) + \
#               tf.mul(Ic, tf.select(tf.logical_and(cond_b, cond_c), wc, zero_f)) + \
#               tf.mul(Id, tf.select(tf.logical_and(cond_b, cond_d), wd, zero_f))

#         # condition = 
#         # img_1 = tf.select(condition, img_1, tf.zeros_like(img_1))
#         # channel.append(img)
#         channel.append(img)
#     # batch.append(tf.pack(channel, axis=1))
#     batch.append(tf.pack(channel, axis=1))
# # preds = tf.pack(batch)
# reconstructs = tf.pack(batch)

# # L1 loss, also try SSIM loss

# # loss_reconstruct = tf.contrib.losses.sum_of_squares(reconstructs, inputs_flat)
# # slim.losses.add_loss(tf.minimum(loss_predict, loss_reconstruct))

# # Charbonnier penalty function
# # loss_min = tf.minimum(loss_predict, loss_reconstruct)
# beta = 255.0
# # diff_predict = tf.mul(tf.sub(preds, outputs_flat), beta)
# diff_reconstruct = tf.mul(tf.sub(reconstructs, inputs_flat), beta)
# # print diff_reconstruct.get_shape()
# # Charbonnier_predict = tf.reduce_mean(tf.pow(tf.square(diff_predict) + tf.square(epsilon), alpha_c))
# Charbonnier_reconstruct = tf.reduce_mean(tf.pow(tf.square(diff_reconstruct) + tf.square(epsilon), alpha_c))


