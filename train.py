import os,sys
from sintelLoader import sintelLoader
from model import deep3D
import tensorflow as tf
import numpy as np
import logging
import subprocess

class train:
    '''Pipeline for training
    '''

    def __init__(self, data_path, image_size, split, passKey):
        self.image_size = image_size
        self.sintel = sintelLoader(data_path, image_size, split, passKey)
        self.batch_size = 1
        self.maxEpochs = 10
        self.maxIterPerEpoch = 1000

        self.trainNet(self.batch_size)
        

    def trainNet(self, batch_size):
        sess = tf.Session()
        imgs = tf.placeholder(tf.float32, [batch_size, self.image_size[0], self.image_size[1], 3])
        # The 'vgg16_weights.npz' file can be downloaded at http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
        vgg16 = 'vgg16_weights.npz'
        if not os.path.exists(vgg16):
            print "Downloading the pre-trained VGG16 model"
            try:
                subprocess.call("wget http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz", shell=True)
            except:
                print 'Something wrong during the downloading.'
        deep3D_model = deep3D(imgs, vgg16, sess)
        print('Finished building Network.')

        logging.info("Start Initializing Variabels.")
        # What about pre-traind initialized model params and deconv parms? 
        init = tf.initialize_all_variables()
        sess.run(tf.initialize_all_variables())

        sys.exit()
        for epoch in xrange(1, self.maxEpochs+1):
            print("Epoch %d: \r\n" % epoch)
            for iteration in xrange(1, self.maxIterPerEpoch+1):
                _, loss_value = self.trainBatch(batch_size, sess)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                print("Iter %04d: Loss %2.4f \r\n" % (iteration, loss_value))

                
    def trainBatch(self, batch_size, sess):
        source, target = self.sintel.sampleTrain(self.batch_size)
        source_tf = tf.convert_to_tensor(source)
        target_tf = tf.convert_to_tensor(target)
        print source_tf.get_shape()



