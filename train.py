import os,sys
from sintelLoader import sintelLoader
import warpFlow
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import logging
import subprocess
from PIL import Image


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('overwrite', True, 'Overwrite existing directory.')

tf.app.flags.DEFINE_integer('save_summaries_secs', 10,
                     'The frequency of which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer('save_interval_secs', 3000,
                     'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000,
                     'The maximum number of gradient steps.')

tf.app.flags.DEFINE_float('learning_rate', .001, 'The learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 10,
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
        self.maxEpochs = 10
        self.maxIterPerEpoch = 1000

        self.trainNet(self.batch_size)
        

    def trainNet(self, batch_size):

	if not os.path.isdir(FLAGS.train_log_dir):
    		os.makedirs(FLAGS.train_log_dir, mode=0777)

	sess = tf.Session()
        source_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        loss_val,  preds  = warpFlow.Model(source_img, target_img)
        print('Finished building Network.')

        logging.info("Start Initializing Variabels.")
        # What about pre-traind initialized model params and deconv parms? 
        init = tf.initialize_all_variables()
	total_loss = slim.losses.get_total_loss()
   	train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)

        sess.run(tf.initialize_all_variables())
        for epoch in xrange(1, self.maxEpochs+1):
            print("Epoch %d: \r\n" % epoch)
            for iteration in xrange(1, self.maxIterPerEpoch+1):
            	source, target = self.sintel.sampleTrain(self.batch_size)
		if iteration%10 == 0:
			loss_values, predictions = sess.run([loss_val, preds], feed_dict={source_img: source, target_img: target})
			flowx = Image.fromarray(np.squeeze(predictions[0, :, :, 0]))
			flowx = flowx.convert("RGB")
			flowy = Image.fromarray(np.squeeze(predictions[0, :, :, 1]))
			flowy = flowy.convert("RGB")
			flowx.save(FLAGS.train_log_dir + "flowx_" + str(iteration) + ".jpeg")
			flowy.save(FLAGS.train_log_dir + "flowy_" + str(iteration) + ".jpeg")
			assert not np.isnan(loss_values), 'Model diverged with loss = NaN'
	             	print("Iter %04d: Loss %2.4f \r\n" % (iteration, loss_values))
		train_op.run(feed_dict={source_img: source, target_img: target}, session=sess)

