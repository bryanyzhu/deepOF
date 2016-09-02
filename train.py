import os,sys
from sintelLoader import sintelLoader
import warpFlow
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import logging
import subprocess


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of images in each batch.')

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

  	g = tf.Graph()
  	with g.as_default():
        	source, target = self.sintel.sampleTrain(self.batch_size)
        	source_tf = tf.convert_to_tensor(source)
        	target_tf = tf.convert_to_tensor(target)

        	preds = warpFlow.Model(source_tf, target_tf)

		total_loss = slim.losses.get_total_loss()

    		# Configure the learning rate using an exponetial decay.
    		decay_steps = int(len(self.sintel.trainList) / FLAGS.batch_size *
                 	     FLAGS.num_epochs_per_decay)

   		learning_rate = tf.train.exponential_decay(
   		    FLAGS.learning_rate,
   		    tf.Variable(1, trainable=False),
   		    decay_steps,
   		    FLAGS.learning_rate_decay_factor,
   		    staircase=True)

   		# Specify the optimization scheme:
   		# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
   		optimizer = tf.train.AdamOptimizer(learning_rate)

   		# Set up the training tensor:
		train_op = slim.learning.create_train_op(total_loss, optimizer)

    		tf.scalar_summary('Total Loss', total_loss)
    		tf.scalar_summary('Learning rate', learning_rate)
    		tf.image_summary('inputs', tf.expand_dims(source_tf[0, :, :, :], 0))
    		tf.image_summary('predictions', tf.expand_dims(preds[0, :, :, :], 0))
    		tf.image_summary('ground_truth', tf.expand_dims(target_tf[0, :, :, :], 0))

   		 # Run training.
   		slim.learning.train(train_op,
   		                     FLAGS.train_log_dir,
   		                     number_of_steps=FLAGS.max_number_of_steps,
				     graph = g, 
   		                     save_summaries_secs=FLAGS.save_summaries_secs,
   		                     save_interval_secs=FLAGS.save_interval_secs)



