import os,sys
from flyingChairsLoader import flyingChairsLoader
import flyingChairsWrapFlow_vgg
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import subprocess
import cv2
import math
import flyingChairsUtils as utils


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/fc_vgg_aug_v3/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 8, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('overwrite', True, 'Overwrite existing directory.')

tf.app.flags.DEFINE_integer('save_interval_epoch', 18,
                     'The frequency with which the model is saved, in epoch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000,
                     'The maximum number of gradient steps.')

tf.app.flags.DEFINE_float('learning_rate', 0.000016, 'The learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                   """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 18,
                   """Number of epochs after which learning rate decays.""")

tf.app.flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
FLAGS = tf.app.flags.FLAGS



class train:
    '''Pipeline for training
    '''

    def __init__(self, data_path, image_size):
        self.image_size = image_size
        self.origin_size = [384, 512]
        self.numLosses = 5
        self.lambda_smooth = 1.0
        self.mean = np.array([97.533268117955444, 99.238235788550085, 97.055973199626948], dtype=np.float32)
        self.flyingChairs = flyingChairsLoader(data_path, self.image_size)
        self.batch_size = FLAGS.batch_size
        self.maxEpochs = 110
        self.maxIterPerEpoch = int(math.floor(len(self.flyingChairs.trainList)/self.batch_size))
        print("One epoch has %d iterations based on batch size %d." % (self.maxIterPerEpoch, self.batch_size))
        self.trainNet(self.batch_size)

    def downloadModel(self, modelUrl):
        subprocess.call(["wget %s" % modelUrl], shell=True)

    def load_VGG16_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        cutLayerNum = []
        offNum = 0
        for i, k in enumerate(keys):
            if i <= 25:
                if i in cutLayerNum:
                    offNum += 1
                    print i, k, np.shape(weights[k]), "not included in our flow model"
                else:
                    if i == 0:
                        sess.run(self.VGG_init_vars[i-offNum].assign(np.repeat(weights[k],2,axis=2)))
                        print i, k, np.shape(np.repeat(weights[k],2,axis=2))
                    else:
                        sess.run(self.VGG_init_vars[i-offNum].assign(weights[k]))
                        print i, k, np.shape(weights[k])

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

        # Figure out the dimension information
        source, target, _ = self.flyingChairs.sampleTrain(self.batch_size, 1)
        source_geo, target_geo = utils.geoAugmentation(source, target)
        dimen_info = source_geo.shape

        with tf.device('/gpu:0'):
            photo_source = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            photo_target = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            geo_source = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            geo_target = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            loss_weight = tf.placeholder(tf.float32, [self.numLosses])
            loss, midFlows, previous = flyingChairsWrapFlow_vgg.VGG16(photo_source, photo_target, geo_source, geo_target, loss_weight)
            print('Finished building Network.')

            # Calculating the number of params inside a network
            model_vars = tf.trainable_variables()
            total_parameters = 0
            for varCount in model_vars:
                # shape is an array of tf.Dimension
                shape = varCount.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            print("Our VGG flownet has %4.2fM number of parameters. " % (total_parameters/1000000.0))

            init = tf.initialize_all_variables()
            total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
            lr = FLAGS.learning_rate
            learning_rate = tf.placeholder(tf.float32, shape=[])
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)
            sess.run(tf.initialize_all_variables())
 
            self.VGG_init_vars = [var for var in model_vars if (var.name).startswith('conv')]
            self.deconv_bilinearInit_vars = [var for var in model_vars if (var.name).startswith('up')]  

            VGG16Init = False           # True, will make optical flow very large, will try later
            bilinearInit = True

            # Use pre-trained VGG16 model to initialize conv filters
            if VGG16Init:
                VGG16modelPath = "/home/yzhu25/Documents/back_deepOF/vgg16_weights.npz"
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
            
            display = 5        # number of iterations to display training log
            # Loss weights schedule
            weight_L = [16,8,4,2,1]
            for epoch in xrange(1, self.maxEpochs+1):
                print("Epoch %d: \r\n" % epoch)
                print("Learning Rate %f: \r\n" % lr)
               
                # 5558 max iterations
                # print self.maxIterPerEpoch
                for iteration in xrange(1, self.maxIterPerEpoch+1):
                    source, target, _ = self.flyingChairs.sampleTrain(self.batch_size, iteration)  

                    # Pre-scaling
                    source = (source - self.mean)/255.0
                    target = (target - self.mean)/255.0

                    # Augmentation
                    source_geo, target_geo = utils.geoAugmentation(source, target)
                    source_photo, target_photo = utils.photoAugmentation(source_geo, target_geo, self.mean)
                    
                    # Training
                    train_op.run(feed_dict = {photo_source: source_photo, 
                                              photo_target: target_photo,
                                              geo_source: source_geo,
                                              geo_target: target_geo, 
                                              loss_weight: weight_L, 
                                              learning_rate: lr}, session = sess)
                    
                    if iteration % display == 0:
                        losses, flows_all, loss_sum = sess.run([loss, midFlows, total_loss], 
                                 feed_dict = {photo_source: source_photo, 
                                              photo_target: target_photo,
                                              geo_source: source_geo,
                                              geo_target: target_geo, 
                                              loss_weight: weight_L})
                
                        print("---Train Batch(%d): Epoch %03d Iter %04d: Loss_sum %4.4f \r\n" % (self.batch_size, epoch, iteration, loss_sum))
                        print("          PhotometricLoss1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["Charbonnier_reconstruct"], weight_L[0], losses[0]["Charbonnier_reconstruct"] * weight_L[0]))
                        print("          PhotometricLoss2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["Charbonnier_reconstruct"], weight_L[1], losses[1]["Charbonnier_reconstruct"] * weight_L[1]))
                        print("          PhotometricLoss3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["Charbonnier_reconstruct"], weight_L[2], losses[2]["Charbonnier_reconstruct"] * weight_L[2]))
                        print("          PhotometricLoss4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["Charbonnier_reconstruct"], weight_L[3], losses[3]["Charbonnier_reconstruct"] * weight_L[3]))
                        print("          PhotometricLoss5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["Charbonnier_reconstruct"], weight_L[4], losses[4]["Charbonnier_reconstruct"] * weight_L[4]))
                        print("          SmoothnessLossU1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["U_loss"], weight_L[0]*self.lambda_smooth, losses[0]["U_loss"] * weight_L[0]*self.lambda_smooth))
                        print("          SmoothnessLossU2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["U_loss"], weight_L[1]*self.lambda_smooth, losses[1]["U_loss"] * weight_L[1]*self.lambda_smooth))
                        print("          SmoothnessLossU3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["U_loss"], weight_L[2]*self.lambda_smooth, losses[2]["U_loss"] * weight_L[2]*self.lambda_smooth))
                        print("          SmoothnessLossU4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["U_loss"], weight_L[3]*self.lambda_smooth, losses[3]["U_loss"] * weight_L[3]*self.lambda_smooth))
                        print("          SmoothnessLossU5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["U_loss"], weight_L[4]*self.lambda_smooth, losses[4]["U_loss"] * weight_L[4]*self.lambda_smooth))
                        print("          SmoothnessLossV1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["V_loss"], weight_L[0]*self.lambda_smooth, losses[0]["V_loss"] * weight_L[0]*self.lambda_smooth))
                        print("          SmoothnessLossV2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["V_loss"], weight_L[1]*self.lambda_smooth, losses[1]["V_loss"] * weight_L[1]*self.lambda_smooth))
                        print("          SmoothnessLossV3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["V_loss"], weight_L[2]*self.lambda_smooth, losses[2]["V_loss"] * weight_L[2]*self.lambda_smooth))
                        print("          SmoothnessLossV4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["V_loss"], weight_L[3]*self.lambda_smooth, losses[3]["V_loss"] * weight_L[3]*self.lambda_smooth))
                        print("          SmoothnessLossV5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["V_loss"], weight_L[4]*self.lambda_smooth, losses[4]["V_loss"] * weight_L[4]*self.lambda_smooth))
                        
                        assert not np.isnan(loss_sum).any(), 'Model diverged with loss = NaN'
                    if iteration == self.maxIterPerEpoch:   
                        print("Start evaluating......")
                        self.evaluateNet(epoch, dimen_info[0], weight_L, sess, 
                            loss, midFlows, previous, photo_source, photo_target, geo_source, geo_target, loss_weight)
                    
                if epoch % FLAGS.num_epochs_per_decay == 0:
                    lr *= FLAGS.learning_rate_decay_factor

                if epoch % FLAGS.save_interval_epoch == 0:
                    print("Save to " +  FLAGS.train_log_dir + str(epoch) + '_model.ckpt')
                    saver.save(sess, FLAGS.train_log_dir + str(epoch) + '_model.ckpt')


    def evaluateNet(self, 
                    epoch, 
                    batch_size, 
                    weight_L, 
                    sess, 
                    loss, 
                    midFlows, 
                    previous,
                    photo_source, 
                    photo_target, 
                    geo_source, 
                    geo_target, 
                    loss_weight):

        testBatchSize = batch_size
        maxTestIter = int(math.floor(len(self.flyingChairs.valList)/testBatchSize))
        Loss1, Loss2, Loss3, Loss4, Loss5 = 0,0,0,0,0
        U_Loss1, U_Loss2, U_Loss3, U_Loss4, U_Loss5 = 0,0,0,0,0
        V_Loss1, V_Loss2, V_Loss3, V_Loss4, V_Loss5 = 0,0,0,0,0
        flow_1 = []
        flow_gt = []
        previous_img = []
        for iteration in xrange(1, maxTestIter+1):
            testBatch = self.flyingChairs.sampleVal(testBatchSize, iteration)
            source, target, flow = testBatch[0]
            # pre-scaling
            source = (source - self.mean)/255.0
            target = (target - self.mean)/255.0

            imgPath = testBatch[1][0]
            losses, flows_all, prev_all = sess.run([loss, midFlows, previous], 
                                    feed_dict = {photo_source: source, 
                                                 photo_target: target,
                                                 geo_source: source,
                                                 geo_target: target, 
                                                 loss_weight: weight_L})
            Loss1 += losses[0]["total"]
            Loss2 += losses[1]["total"]
            Loss3 += losses[2]["total"]
            Loss4 += losses[3]["total"]
            Loss5 += losses[4]["total"]
            U_Loss1 += losses[0]["U_loss"]
            U_Loss2 += losses[1]["U_loss"]
            U_Loss3 += losses[2]["U_loss"]
            U_Loss4 += losses[3]["U_loss"]
            U_Loss5 += losses[4]["U_loss"]
            V_Loss1 += losses[0]["V_loss"]
            V_Loss2 += losses[1]["V_loss"]
            V_Loss3 += losses[2]["V_loss"]
            V_Loss4 += losses[3]["V_loss"]
            V_Loss5 += losses[4]["V_loss"]

            flow1_list = []
            previous_img_list = []
            for batch_idx in xrange(testBatchSize):
                flowImg = flows_all[0][batch_idx,:,:,:]*2       # pr1 is still half of the final predicted flow value
                flowImg = np.clip(flowImg, -204.4790, 201.3478)       # 300 and 250 is the min and max of the flow value in training dataset
                flow1_list.append(np.expand_dims(cv2.resize(flowImg, (self.origin_size[1], self.origin_size[0])), 0))
                previous_img_list.append(np.expand_dims(cv2.resize(prev_all[batch_idx,:,:,:], (self.origin_size[1], self.origin_size[0])), 0))
            flow_1.append(np.concatenate(flow1_list, axis=0))
            previous_img.append(np.concatenate(previous_img_list, axis=0))
            flow_gt.append(flow)

            # Visualize
            # if iteration % 2 == 0:
            if epoch == 1:      # save ground truth images and flow
                source = source * 255.0 + self.mean
                target = target * 255.0 + self.mean
                source = np.clip(source, 0, 255)
                target = np.clip(target, 0, 255)
                source = source.astype(int)
                target = target.astype(int)
                gt_1 = source[0, :, :, :].squeeze()
                cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_img1.jpeg", gt_1)
                gt_2 = target[0, :, :, :].squeeze()
                cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_img2.jpeg", gt_2)
                GTflowColor = utils.flowToColor(flow[0,:,:,:].squeeze())
                cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_gt_flow.jpeg", GTflowColor)

            flowColor_1 = utils.flowToColor(flow_1[iteration-1][0,:,:,:].squeeze())
            # print flowColor.max(), flowColor.min(), flowColor.mean()
            cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_flowColor_1" + ".jpeg", flowColor_1)

            prev_frame = previous_img[iteration-1][0,:,:,:]
            intensity_range = np.max(prev_frame, axis=None) - np.min(prev_frame, axis=None)
            # save predicted next frames
            prev_frame = (prev_frame - np.min(prev_frame, axis=None)) * 255 / intensity_range
            cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_prev_1" + ".jpeg", prev_frame.astype(int))

        # Calculate endpoint error
        f1 = np.concatenate(flow_1, axis=0)
        f2 = np.concatenate(flow_gt, axis=0)
        AEE = utils.flow_ee(f1, f2)

        # calculate statistics
        if epoch == 1:
            print("***Test: max (flow_gt) %2.4f  min (flow_gt) %2.4f  abs_mean (flow_gt) %2.4f \r\n"
                % (np.amax(f2, axis=None), np.amin(f2, axis=None), np.mean(np.absolute(f2), axis=None)))
        print("***Test flow abs_mean: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.mean(np.absolute(flows_all[0]), axis=None), np.mean(np.absolute(flows_all[1]), axis=None), np.mean(np.absolute(flows_all[2]), axis=None), 
                np.mean(np.absolute(flows_all[3]), axis=None), np.mean(np.absolute(flows_all[4]), axis=None)))
        print("***Test flow max: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.max(np.absolute(flows_all[0]), axis=None), np.max(np.absolute(flows_all[1]), axis=None), np.max(np.absolute(flows_all[2]), axis=None), 
                np.max(np.absolute(flows_all[3]), axis=None), np.max(np.absolute(flows_all[4]), axis=None)))
        Loss_sum = (Loss1*weight_L[0] + Loss2*weight_L[1] + Loss3*weight_L[2] + Loss4*weight_L[3] + Loss5*weight_L[4])/maxTestIter
        ULoss_sum = (U_Loss1*weight_L[0] + U_Loss2*weight_L[1] + U_Loss3*weight_L[2] + U_Loss4*weight_L[3] + U_Loss5*weight_L[4])/maxTestIter*self.lambda_smooth
        VLoss_sum = (V_Loss1*weight_L[0] + V_Loss2*weight_L[1] + V_Loss3*weight_L[2] + V_Loss4*weight_L[3] + V_Loss5*weight_L[4])/maxTestIter*self.lambda_smooth
        print("***Test: Epoch %03d : Loss_sum %4.4f ULoss_sum %4.4f VLoss_sum %4.4f AEE %4.4f \r\n" 
            % (epoch, Loss_sum, ULoss_sum, VLoss_sum, AEE))