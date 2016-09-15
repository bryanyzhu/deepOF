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
import utils as utils


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/data_1/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 8, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('overwrite', True, 'Overwrite existing directory.')

tf.app.flags.DEFINE_integer('save_interval_epoch', 10,
                     'The frequency with which the model is saved, in epoch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000,
                     'The maximum number of gradient steps.')

tf.app.flags.DEFINE_float('learning_rate', .00001, 'The learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
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

    def downloadModel(self, modelUrl):
        subprocess.call(["wget %s" % modelUrl], shell=True)

    def load_VGG16_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        cutLayerNum = [2,3,6,7,12,13,18,19,24,25]
        offNum = 0
        for i, k in enumerate(keys):
            if i <= 25:
                if i in cutLayerNum:
                    offNum += 1
                    # print i, k, np.shape(weights[k]), "not included in deep3D model"
                else:
                    # print i, k, np.shape(weights[k])
                    if i == 0:
                        sess.run(self.VGG_init_vars[i-offNum].assign(np.repeat(weights[k],2,axis=2)))
                    else:
                        sess.run(self.VGG_init_vars[i-offNum].assign(weights[k]))

    def load_deconv_weights(self, var, sess):
        f_shape = sess.run(var).shape
        width = f_shape[0]
        heigh = f_shape[0]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        sess.run(var.assign(weights))        

    def trainNet(self, batch_size):

        if not os.path.isdir(FLAGS.train_log_dir):
            os.makedirs(FLAGS.train_log_dir, mode=0777)

        sess = tf.Session()
        source_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        loss_p,  loss_r, flow_pred, frame_pred  = wrapFlow.Model(source_img, target_img)
        print('Finished building Network.')

        logging.info("Start Initializing Variabels.")
        init = tf.initialize_all_variables()
        total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
    
        lr = FLAGS.learning_rate
        learning_rate = tf.placeholder(tf.float32, shape=[])
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

        sess.run(tf.initialize_all_variables())
        # What about pre-traind initialized model params and deconv parms? 
        model_vars = tf.trainable_variables()
        self.VGG_init_vars = [var for var in model_vars if (var.name).startswith('conv')]
        self.deconv_bilinearInit_vars = [var for var in model_vars if (var.name).startswith('deconv')]  

        VGG16Init = False
        bilinearInit = False

        # Use pre-trained VGG16 model to initialize conv filters
        if VGG16Init:
            VGG16modelPath = "vgg16_weights.npz"
            if not os.path.exists(VGG16modelPath):
                modelUrl = "http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz"
                self.downloadModel(modelUrl)
            self.load_VGG16_weights(VGG16modelPath, sess)
            print("-----Done initializing conv filters with VGG16 pre-trained model------")

        # Use bilinear upsampling to initialize deconv filters
        if bilinearInit:
            for var in self.deconv_bilinearInit_vars:
                if "weights" in var.name:
                    self.load_deconv_weights(var, sess)
            print("-----Done initializing deconv filters with bilinear upsampling------")
    
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Restore from " +  ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        display = 20        # number of iterations to display training log
        for epoch in xrange(1, self.maxEpochs+1):
            print("Epoch %d: \r\n" % epoch)
            print("Learning Rate %f: \r\n" % lr)

            for iteration in xrange(1, self.maxIterPerEpoch+1):
                # source, target = self.sintel.sampleTrain(self.batch_size, iteration)      # last several images will be missing
                source, target, flow = self.sintel.sampleTrain(self.batch_size)
                train_op.run(feed_dict = {source_img: source, target_img: target, learning_rate: lr}, session = sess)
                if iteration % display == 0:
                # if iteration == 1:
                    loss_ps, loss_rs, flow_preds, frame_preds = sess.run([loss_p, loss_r, flow_pred, frame_pred], feed_dict={source_img: source, target_img: target})
                    flow_error = ((flow_preds - flow) ** 2).mean(axis=None)/self.batch_size
                    assert not np.isnan(loss_ps) and not np.isnan(loss_rs), 'Model diverged with loss = NaN'
                    print("---Training: Epoch %03d Iter %04d: Prediction Loss %2.4f Reconstruction Loss %2.4f Flow MSE %4.4f \r\n" 
                        % (epoch, iteration, loss_ps, loss_rs, flow_error))
                if iteration == int(math.floor(self.maxIterPerEpoch/2)) or iteration == self.maxIterPerEpoch:
                # if True:
                    self.evaluateNet(epoch, iteration, sess)

            if epoch % FLAGS.num_epochs_per_decay == 0:
                lr *= FLAGS.learning_rate_decay_factor

            if epoch % FLAGS.save_interval_epoch == 0:
                print("Save to " +  FLAGS.train_log_dir + str(epoch) + '_model.ckpt')
                saver.save(sess, FLAGS.train_log_dir + str(epoch) + '_model.ckpt')


    def evaluateNet(self, epoch, trainIter, sess):
        # For Sintel, the batch size should be 7, so that all validation images are covered.
        testBatchSize = 7
        source_img = tf.placeholder(tf.float32, [testBatchSize, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [testBatchSize, self.image_size[0], self.image_size[1], 3])

        # Don't know if this is safe to set all variables reuse=True
        # But because of different batch size, I don't know how to evaluate the model on validation data
        tf.get_variable_scope().reuse_variables()

        loss_p, loss_r, flow_pred, frame_pred  = wrapFlow.Model(source_img, target_img)
        maxTestIter = int(math.floor(len(self.sintel.valList)/testBatchSize))
        totalPLoss = 0
        totalRLoss = 0
        flow_p = []
        flow_gt = []
        for iteration in xrange(1, maxTestIter+1):
            testBatch = self.sintel.sampleVal(testBatchSize, iteration)
            source, target, flow = testBatch[0]
            imgPath = testBatch[1][0]
            loss_ps, loss_rs, flow_preds, frame_preds = sess.run([loss_p, loss_r, flow_pred, frame_pred], feed_dict={source_img: source, target_img: target})
            # assert not np.isnan(loss_values), 'Model diverged with loss = NaN'
            totalPLoss += loss_ps
            totalRLoss += loss_rs
            flow_p.append(flow_preds)
            flow_gt.append(flow)

            # Visualize
            if iteration % 4 == 0:
                if epoch == 1:
                    gt_1 = source[0, :, :, :].squeeze()
                    cv2.imwrite(FLAGS.train_log_dir + self.sintel.valList[imgPath][0].replace("/", "-")[:-4] + ".jpeg", gt_1)
                    gt_2 = target[0, :, :, :].squeeze()
                    cv2.imwrite(FLAGS.train_log_dir + self.sintel.valList[imgPath][1].replace("/", "-")[:-4] + ".jpeg", gt_2)
                    GTflowColor = utils.flowToColor(flow[0,:,:,:].squeeze())
                    cv2.imwrite(FLAGS.train_log_dir + self.sintel.valList[imgPath][0].replace("/", "-")[:-4] + "_gt_flow.jpeg", GTflowColor)

                # flowx = np.squeeze(flow_preds[0, :, :, 0])
                # flowx = (flowx-flowx.min())/(flowx.max()-flowx.min())
                # flowx = np.uint8(flowx*255.0)
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowx" + ".jpeg", flowx)
                # flowy = np.squeeze(flow_preds[0, :, :, 1])
                # flowy = (flowy-flowy.min())/(flowy.max()-flowy.min())
                # flowy = np.uint8(flowy*255.0)
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowy" + ".jpeg", flowy)

                flowColor = utils.flowToColor(flow_preds[0,:,:,:].squeeze())
                # print flowColor.max(), flowColor.min(), flowColor.mean()
                cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor" + ".jpeg", flowColor)

        # Calculate endpoint error
        AEE = utils.flow_ee(np.concatenate(flow_p, axis=0), np.concatenate(flow_gt, axis=0))
        # EE_tot, AEE = utils.flow_ee(np.concatenate(flow_p, axis=0), np.concatenate(flow_gt, axis=0))

        # Calculate anguar error, maybe not necessary
        AAE = utils.flow_ae(np.concatenate(flow_p, axis=0), np.concatenate(flow_gt, axis=0))
        # AE_tot, AAE = utils.flow_ae(np.concatenate(flow_p, axis=0), np.concatenate(flow_gt, axis=0))

        print("***Test: Epoch %03d Iter %04d: Pred Loss %2.4f Recons Loss %2.4f AEE %4.4f AAE %4.4f \r\n" 
            % (epoch, trainIter, totalPLoss/maxTestIter, totalRLoss/maxTestIter, AEE, AAE))

        



            # flowx = np.squeeze(flow_preds[0, :, :, 0])
            # print flowx.min(), flowx.max()
            # flowx = (flowx-flowx.min())/(flowx.max()-flowx.min())
            # flowx = np.uint8(flowx*255.0)
            # flowx = Image.fromarray(flowx)
            # flowx = flowx.convert("RGB")
            # flowy = np.squeeze(flow_preds[0, :, :, 1])
            # print flowy.min(), flowy.max()
            # flowy = (flowy-flowy.min())/(flowy.max()-flowy.min())
            # flowy = np.uint8(flowy*255.0)
            # flowy = Image.fromarray(flowy)
            # flowy = flowy.convert("RGB")

            # pred = np.squeeze(frame_preds[0, :, :, :])
            # print pred.min(), pred.max()
            # pred = np.uint8(pred)
            # pred = Image.fromarray(pred)
            # pred = pred.convert("RGB")

            # flowx.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_flowx" + ".jpeg")
            # flowy.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_flowy" + ".jpeg")
            # pred.save(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_pred" + ".jpeg")

            # gt_1 = source[0, :, :, :].squeeze()
            # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_gt1" + ".jpeg", gt_1)
            # gt_2 = target[0, :, :, :].squeeze()
            # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_gt2" + ".jpeg", gt_2)


            
            # print("Iter %04d: Loss %2.4f \r\n" % (iteration, loss_values))