import os,sys
from sintelLoader import sintelLoader
import wrapFlow
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import logging
import subprocess
from PIL import Image
import cv2
import math


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 10, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('overwrite', True, 'Overwrite existing directory.')


tf.app.flags.DEFINE_integer('save_interval_epoch', 10,
                     'The frequency with which the model is saved, in epoch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000,
                     'The maximum number of gradient steps.')

tf.app.flags.DEFINE_float('learning_rate', .001, 'The learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 40,
                   """Number of epochs after which learning rate decays.""")

tf.app.flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
FLAGS = tf.app.flags.FLAGS



class train:
    '''Pipeline for training
    '''

    def __init__(self, data_path, image_size, split, passKey):
        self.image_size = image_size
        self.sintel = sintelLoader(data_path, image_size, split, passKey)
        self.batch_size = FLAGS.batch_size
        self.maxEpochs = 1000
        self.maxIterPerEpoch = int(math.floor(len(self.sintel.trainList)/self.batch_size))

        self.trainNet(self.batch_size)
        

    def trainNet(self, batch_size):

	if not os.path.isdir(FLAGS.train_log_dir):
    		os.makedirs(FLAGS.train_log_dir, mode=0777)

	sess = tf.Session()
        source_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        loss_val,  flow_pred, frame_pred  = wrapFlow.Model(source_img, target_img)
        print('Finished building Network.')

        logging.info("Start Initializing Variabels.")
        # What about pre-traind initialized model params and deconv parms? 
        init = tf.initialize_all_variables()
	total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
	
	lr = FLAGS.learning_rate
	learning_rate = tf.placeholder(tf.float32, shape=[])
   	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

        sess.run(tf.initialize_all_variables())
	
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_log_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("Restore from " +  ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	
        for epoch in xrange(1, self.maxEpochs+1):
            print("Epoch %d: \r\n" % epoch)
            print("Learning Rate %f: \r\n" % lr)
            for iteration in xrange(1, self.maxIterPerEpoch+1):
            	source, target = self.sintel.sampleTrain(self.batch_size, iteration)
		if iteration%40 == 0:
			loss_values, flow_preds, frame_preds = sess.run([loss_val, flow_pred, frame_pred], feed_dict={source_img: source, target_img: target})
			flowx = np.squeeze(flow_preds[0, :, :, 0])
			print flowx.min(), flowx.max()
			flowx = (flowx-flowx.min())/(flowx.max()-flowx.min())
			flowx = np.uint8(flowx*255.0)
			flowx = Image.fromarray(flowx)
			flowx = flowx.convert("RGB")
			flowy = np.squeeze(flow_preds[0, :, :, 1])
			print flowy.min(), flowy.max()
			flowy = (flowy-flowy.min())/(flowy.max()-flowy.min())
			flowy = np.uint8(flowy*255.0)
			flowy = Image.fromarray(flowy)
			flowy = flowy.convert("RGB")
			
			pred = np.squeeze(frame_preds[0, :, :, :])
			print pred.min(), pred.max()
			pred = np.uint8(pred)
			pred = Image.fromarray(pred)
			pred = pred.convert("RGB")

			flowx.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_flowx" + ".jpeg")
                        flowy.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_flowy" + ".jpeg")
                        pred.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_pred" + ".jpeg")

                    	gt_1 = source[0, :, :, :].squeeze()
                    	cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_gt1" + ".jpeg", gt_1)
                    	gt_2 = target[0, :, :, :].squeeze()
                    	cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_gt2" + ".jpeg", gt_2)


			assert not np.isnan(loss_values), 'Model diverged with loss = NaN'
	             	print("Iter %04d: Loss %2.4f \r\n" % (iteration, loss_values))
		train_op.run(feed_dict={source_img: source, target_img: target, learning_rate: lr}, session=sess)

	    if epoch % FLAGS.num_epochs_per_decay==0:
		lr *= FLAGS.learning_rate_decay_factor

	    if epoch%FLAGS.save_interval_epoch == 0:
		print("Save to " +  FLAGS.train_log_dir + str(epoch) + '_model.ckpt')
		saver.save(sess, FLAGS.train_log_dir + str(epoch) + '_model.ckpt')
