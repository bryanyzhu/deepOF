import os,sys
from flyingChairsLoader import flyingChairsLoader
import flyingChairsWrapFlow
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import subprocess
import cv2
import math
import utils as utils


tf.app.flags.DEFINE_string('train_log_dir', '/tmp/trial_1/',
                    'Directory where to write event logs.')

tf.app.flags.DEFINE_integer('batch_size', 4, 'The number of images in each batch.')

tf.app.flags.DEFINE_integer('overwrite', True, 'Overwrite existing directory.')

tf.app.flags.DEFINE_integer('save_interval_epoch', 30,
                     'The frequency with which the model is saved, in epoch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000000,
                     'The maximum number of gradient steps.')

tf.app.flags.DEFINE_float('learning_rate', 0.000016, 'The learning rate')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                   """Learning rate decay factor.""")

tf.app.flags.DEFINE_float('num_epochs_per_decay', 30,
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
        self.numLosses = 6
        self.lambda_smooth = 0.5
        self.flyingChairs = flyingChairsLoader(data_path, self.image_size)
        self.batch_size = FLAGS.batch_size
        self.maxEpochs = 110
        self.maxIterPerEpoch = int(math.floor(len(self.flyingChairs.trainList)/self.batch_size))

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
                    print i, k, np.shape(weights[k]), "not included in deep3D model"
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

        # with tf.device('/gpu:1'):
        source_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], self.image_size[1], 3])
        loss_weight = tf.placeholder(tf.float32, [self.numLosses])
        loss, midFlows, previous = flyingChairsWrapFlow.flowNet(source_img, target_img, loss_weight)
        print('Finished building Network.')

        init = tf.initialize_all_variables()
        total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
        lr = FLAGS.learning_rate
        learning_rate = tf.placeholder(tf.float32, shape=[])
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        # What about pre-traind initialized model params and deconv parms? 
        model_vars = tf.trainable_variables()
        self.VGG_init_vars = [var for var in model_vars if (var.name).startswith('conv')]
        self.deconv_bilinearInit_vars = [var for var in model_vars if (var.name).startswith('up')]  

        # # Calculating the number of params inside a network
        # total_parameters = 0
        # for varCount in model_vars:
        #     # shape is an array of tf.Dimension
        #     shape = varCount.get_shape()
        #     # print(shape)
        #     # print(len(shape))
        #     variable_parametes = 1
        #     for dim in shape:
        #         variable_parametes *= dim.value
        #     total_parameters += variable_parametes
        # print("Our FlowNet has %4.2fM number of parameters. " % (total_parameters/1000000.0))

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
        
        display = 1        # number of iterations to display training log
        # Loss weights schedule
        weight_L = [16,8,4,2,1,0.5]
        for epoch in xrange(1, self.maxEpochs+1):
            print("Epoch %d: \r\n" % epoch)
            print("Learning Rate %f: \r\n" % lr)
           
            # 5558 max iterations
            # print self.maxIterPerEpoch
            for iteration in xrange(1, self.maxIterPerEpoch+1):
                source, target, flow = self.flyingChairs.sampleTrain(self.batch_size, iteration)  

                # source_geo, target_geo = self.geoAugmentation(source, target, sess)
                ##################################################################
                # Training
                train_op.run(feed_dict = {source_img: source, target_img: target, loss_weight: weight_L, learning_rate: lr}, session = sess)
                
                if iteration % display == 0:
                # if iteration == 1:
                    
                    losses, flows_all, loss_sum = sess.run([loss, midFlows, total_loss], feed_dict={source_img: source, target_img: target, loss_weight: weight_L})
                    
                    batch_num = 0
                    img1 = source[batch_num,:,:,:]
                    img2 = target[batch_num,:,:,:]
                    print np.mean(img1), np.mean(img2)
                    print np.max(img1), np.max(img2)
                    print np.min(img1), np.min(img2)
                    print np.mean(np.abs(img1)), np.mean(np.abs(img2))

                    print np.mean(img1-img2)
                    #print img1.shape
                    print np.mean(np.power(np.square(img1-img2) + (0.001)**2, 0.25))

                    img11 = losses[-2]
                    img22 = losses[-1]
                    print np.mean(img11), np.mean(img22)
                    print np.max(img11), np.max(img22)
                    print np.min(img11), np.min(img22)
                    print np.mean(img11-img22)
                    print img11.shape

                    #img1 = np.reshape(img1, [-1,1])
                    #print img1[:10]
                    #img11 = np.reshape(losses[-2], [-1,1])
                    #print img11[:10]

                    print np.array_equal(img1, img11.astype(int))
                    print np.array_equal(img2, img22.astype(int))
                    
                    #cv2.imwrite("img1" + ".jpeg", img1)
                    #cv2.imwrite("img2" + ".jpeg", img2)
                    #cv2.imwrite("loss1" + ".jpeg", img11)
                    #cv2.imwrite("loss2" + ".jpeg", img22)

                    # print flows_all[5].shape
                    # print np.max(flows_all[5])
                    # print np.min(flows_all[5])
                    # print np.mean(np.abs(flows_all[5]))

                    # np.save("/home/yzhu25/Documents/deepOF/flows_1.npy", flows_all[0])
                    # np.save("/home/yzhu25/Documents/deepOF/flows_2.npy", flows_all[1])
                    # np.save("/home/yzhu25/Documents/deepOF/flows_3.npy", flows_all[2])
                    # np.save("/home/yzhu25/Documents/deepOF/flows_4.npy", flows_all[3])
                    # np.save("/home/yzhu25/Documents/deepOF/flows_5.npy", flows_all[4])
                    # np.save("/home/yzhu25/Documents/deepOF/flows_6.npy", flows_all[5])
                    # np.save("/home/yzhu25/Documents/deepOF/source.pkl", source)
                    # np.save("/home/yzhu25/Documents/deepOF/target.pkl", target)
                    # np.save("/home/yzhu25/Documents/deepOF/flow_gt.pkl", flow)



                    sys.exit()
                    print("---Train Batch(%d): Epoch %03d Iter %04d: Loss_sum %4.4f \r\n" % (self.batch_size, epoch, iteration, loss_sum))
                    # print("          PhotometricLoss1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["Charbonnier_predict"], weight_L[0], losses[0]["Charbonnier_predict"] * weight_L[0]))
                    # print("          PhotometricLoss2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["Charbonnier_predict"], weight_L[1], losses[1]["Charbonnier_predict"] * weight_L[1]))
                    # print("          PhotometricLoss3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["Charbonnier_predict"], weight_L[2], losses[2]["Charbonnier_predict"] * weight_L[2]))
                    # print("          PhotometricLoss4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["Charbonnier_predict"], weight_L[3], losses[3]["Charbonnier_predict"] * weight_L[3]))
                    # print("          PhotometricLoss5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["Charbonnier_predict"], weight_L[4], losses[4]["Charbonnier_predict"] * weight_L[4]))
                    # print("          PhotometricLoss6 = %4.4f (* %2.4f = %2.4f loss)" % (losses[5]["Charbonnier_predict"], weight_L[5], losses[5]["Charbonnier_predict"] * weight_L[5]))
                    print("          PhotometricLoss1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["Charbonnier_reconstruct"], weight_L[0], losses[0]["Charbonnier_reconstruct"] * weight_L[0]))
                    print("          PhotometricLoss2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["Charbonnier_reconstruct"], weight_L[1], losses[1]["Charbonnier_reconstruct"] * weight_L[1]))
                    print("          PhotometricLoss3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["Charbonnier_reconstruct"], weight_L[2], losses[2]["Charbonnier_reconstruct"] * weight_L[2]))
                    print("          PhotometricLoss4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["Charbonnier_reconstruct"], weight_L[3], losses[3]["Charbonnier_reconstruct"] * weight_L[3]))
                    print("          PhotometricLoss5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["Charbonnier_reconstruct"], weight_L[4], losses[4]["Charbonnier_reconstruct"] * weight_L[4]))
                    print("          PhotometricLoss6 = %4.4f (* %2.4f = %2.4f loss)" % (losses[5]["Charbonnier_reconstruct"], weight_L[5], losses[5]["Charbonnier_reconstruct"] * weight_L[5]))
                    print("          SmoothnessLossU1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["U_loss"], weight_L[0]*self.lambda_smooth, losses[0]["U_loss"] * weight_L[0]*self.lambda_smooth))
                    print("          SmoothnessLossU2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["U_loss"], weight_L[1]*self.lambda_smooth, losses[1]["U_loss"] * weight_L[1]*self.lambda_smooth))
                    print("          SmoothnessLossU3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["U_loss"], weight_L[2]*self.lambda_smooth, losses[2]["U_loss"] * weight_L[2]*self.lambda_smooth))
                    print("          SmoothnessLossU4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["U_loss"], weight_L[3]*self.lambda_smooth, losses[3]["U_loss"] * weight_L[3]*self.lambda_smooth))
                    print("          SmoothnessLossU5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["U_loss"], weight_L[4]*self.lambda_smooth, losses[4]["U_loss"] * weight_L[4]*self.lambda_smooth))
                    print("          SmoothnessLossU6 = %4.4f (* %2.4f = %2.4f loss)" % (losses[5]["U_loss"], weight_L[5]*self.lambda_smooth, losses[5]["U_loss"] * weight_L[5]*self.lambda_smooth))
                    print("          SmoothnessLossV1 = %4.4f (* %2.4f = %2.4f loss)" % (losses[0]["V_loss"], weight_L[0]*self.lambda_smooth, losses[0]["V_loss"] * weight_L[0]*self.lambda_smooth))
                    print("          SmoothnessLossV2 = %4.4f (* %2.4f = %2.4f loss)" % (losses[1]["V_loss"], weight_L[1]*self.lambda_smooth, losses[1]["V_loss"] * weight_L[1]*self.lambda_smooth))
                    print("          SmoothnessLossV3 = %4.4f (* %2.4f = %2.4f loss)" % (losses[2]["V_loss"], weight_L[2]*self.lambda_smooth, losses[2]["V_loss"] * weight_L[2]*self.lambda_smooth))
                    print("          SmoothnessLossV4 = %4.4f (* %2.4f = %2.4f loss)" % (losses[3]["V_loss"], weight_L[3]*self.lambda_smooth, losses[3]["V_loss"] * weight_L[3]*self.lambda_smooth))
                    print("          SmoothnessLossV5 = %4.4f (* %2.4f = %2.4f loss)" % (losses[4]["V_loss"], weight_L[4]*self.lambda_smooth, losses[4]["V_loss"] * weight_L[4]*self.lambda_smooth))
                    print("          SmoothnessLossV6 = %4.4f (* %2.4f = %2.4f loss)" % (losses[5]["V_loss"], weight_L[5]*self.lambda_smooth, losses[5]["V_loss"] * weight_L[5]*self.lambda_smooth))

                    # print("***Test flow abs_mean: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f pr6 %2.4f" 
                    #     % (np.mean(np.absolute(flows_all[0]), axis=None), np.mean(np.absolute(flows_all[1]), axis=None), np.mean(np.absolute(flows_all[2]), axis=None), 
                    #         np.mean(np.absolute(flows_all[3]), axis=None), np.mean(np.absolute(flows_all[4]), axis=None), np.mean(np.absolute(flows_all[5]), axis=None)))
                    # print("***Test flow max: pr1 %2.4f \r\n" % (np.max(np.absolute(flows_all[0]))))
                    # sys.exit()
                    assert not np.isnan(loss_sum).any(), 'Model diverged with loss = NaN'
                    sys.exit()
                # if iteration == int(math.floor(self.maxIterPerEpoch/2)) or iteration == self.maxIterPerEpoch:
                if iteration % (display * 100) == 0:    # iteration == self.maxIterPerEpoch:    # 
                # if True:
                    print("Start evaluating......")
                    self.evaluateNet(epoch, iteration, weight_L, sess)
                
            if epoch % FLAGS.num_epochs_per_decay == 0:
                lr *= FLAGS.learning_rate_decay_factor

            if epoch % FLAGS.save_interval_epoch == 0:
                print("Save to " +  FLAGS.train_log_dir + str(epoch) + '_model.ckpt')
                saver.save(sess, FLAGS.train_log_dir + str(epoch) + '_model.ckpt')


    def evaluateNet(self, epoch, trainIter, weight_L, sess):
        # For Sintel, the batch size should be 7, so that all validation images are covered.
        testBatchSize = 8
        source_img = tf.placeholder(tf.float32, [testBatchSize, self.image_size[0], self.image_size[1], 3])
        target_img = tf.placeholder(tf.float32, [testBatchSize, self.image_size[0], self.image_size[1], 3])
        loss_weight = tf.placeholder(tf.float32, [self.numLosses])
        # Don't know if this is safe to set all variables reuse=True
        # But because of different batch size, I don't know how to evaluate the model on validation data
        tf.get_variable_scope().reuse_variables()
        # sess_evaluate = tf.Session()

        loss, midFlows, prev = flyingChairsWrapFlow.flowNet(source_img, target_img, loss_weight)
        maxTestIter = int(math.floor(len(self.flyingChairs.valList)/testBatchSize))
        Loss1, Loss2, Loss3, Loss4, Loss5, Loss6 = 0,0,0,0,0,0
        U_Loss1, U_Loss2, U_Loss3, U_Loss4, U_Loss5, U_Loss6 = 0,0,0,0,0,0
        V_Loss1, V_Loss2, V_Loss3, V_Loss4, V_Loss5, V_Loss6 = 0,0,0,0,0,0
        flow_1, flow_2, flow_3, flow_4, flow_5, flow_6 = [],[],[],[],[],[]
        flow_gt = []
        next_img = []
        previous_img = []
        # print weight_L
        for iteration in xrange(1, maxTestIter+1):
            testBatch = self.flyingChairs.sampleVal(testBatchSize, iteration)
            source, target, flow = testBatch[0]
            imgPath = testBatch[1][0]
            losses, flows_all, prev_all = sess.run([loss, midFlows, prev], feed_dict={source_img: source, target_img: target, loss_weight: weight_L})
            # assert not np.isnan(loss_values), 'Model diverged with loss = NaN'
            Loss1 += losses[0]["total"]
            Loss2 += losses[1]["total"]
            Loss3 += losses[2]["total"]
            Loss4 += losses[3]["total"]
            Loss5 += losses[4]["total"]
            Loss6 += losses[5]["total"]
            U_Loss1 += losses[0]["U_loss"]
            U_Loss2 += losses[1]["U_loss"]
            U_Loss3 += losses[2]["U_loss"]
            U_Loss4 += losses[3]["U_loss"]
            U_Loss5 += losses[4]["U_loss"]
            U_Loss6 += losses[5]["U_loss"]
            V_Loss1 += losses[0]["V_loss"]
            V_Loss2 += losses[1]["V_loss"]
            V_Loss3 += losses[2]["V_loss"]
            V_Loss4 += losses[3]["V_loss"]
            V_Loss5 += losses[4]["V_loss"]
            U_Loss6 += losses[5]["V_loss"]

            flow1_list, flow2_list, flow3_list, flow4_list, flow5_list, flow6_list = [], [], [], [], [], []
            # next_img_list = []
            previous_img_list = []
            # print next.shape
            for batch_idx in xrange(testBatchSize):
                flow1_list.append(np.expand_dims(cv2.resize(flows_all[0][batch_idx,:,:,:]*2, (self.origin_size[1], self.origin_size[0])), 0))
                # flow2_list.append(np.expand_dims(flows_all[1][batch_idx,:,:,:], 0))
                # flow3_list.append(np.expand_dims(flows_all[2][batch_idx,:,:,:], 0))
                # flow4_list.append(np.expand_dims(flows_all[3][batch_idx,:,:,:], 0))
                # flow5_list.append(np.expand_dims(flows_all[4][batch_idx,:,:,:], 0))
                # flow6_list.append(np.expand_dims(flows_all[5][batch_idx,:,:,:], 0))
                # next_img_list.append(np.expand_dims(cv2.resize(next_all[batch_idx,:,:,:], (self.origin_size[1], self.origin_size[0])), 0))
                previous_img_list.append(np.expand_dims(cv2.resize(prev_all[batch_idx,:,:,:], (self.origin_size[1], self.origin_size[0])), 0))
            flow_1.append(np.concatenate(flow1_list, axis=0))
            # flow_2.append(np.concatenate(flow2_list, axis=0))
            # flow_3.append(np.concatenate(flow3_list, axis=0))
            # flow_4.append(np.concatenate(flow4_list, axis=0))
            # flow_5.append(np.concatenate(flow5_list, axis=0))
            # flow_6.append(np.concatenate(flow6_list, axis=0))
            # next_img.append(np.concatenate(next_img_list, axis=0))
            previous_img.append(np.concatenate(previous_img_list, axis=0))


            # flow_1.append(sess.run(tf.image.resize_bilinear(flows_all[0], self.origin_size)))
            # flow_2.append(sess.run(tf.image.resize_bilinear(flows_all[1], self.origin_size)))
            # flow_3.append(sess.run(tf.image.resize_bilinear(flows_all[2], self.origin_size)))
            # flow_4.append(sess.run(tf.image.resize_bilinear(flows_all[3], self.origin_size)))
            # flow_5.append(sess.run(tf.image.resize_bilinear(flows_all[4], self.origin_size)))
            # flow_6.append(sess.run(tf.image.resize_bilinear(flows_all[5], self.origin_size)))

            flow_gt.append(flow)

            # Visualize
            # if False:
            if iteration % 10 == 0:
                if epoch == 1:
                    gt_1 = source[0, :, :, :].squeeze()
                    cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_img1.jpeg", gt_1)
                    gt_2 = target[0, :, :, :].squeeze()
                    cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_img2.jpeg", gt_2)
                    GTflowColor = utils.flowToColor(flow[0,:,:,:].squeeze())
                    cv2.imwrite(FLAGS.train_log_dir + self.flyingChairs.valList[imgPath] + "_gt_flow.jpeg", GTflowColor)

                flowColor_1 = utils.flowToColor(flow_1[iteration-1][0,:,:,:].squeeze())
                # print flowColor.max(), flowColor.min(), flowColor.mean()
                cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_1" + ".jpeg", flowColor_1)

                # next_frame = next_img[iteration-1][0,:,:,:]
                # intensity_range = np.max(next_frame, axis=None) - np.min(next_frame, axis=None)
                # # save predicted next frames
                # next_frame = (next_frame - np.min(next_frame, axis=None)) * 255 / intensity_range
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_next_1" + ".jpeg", next_frame.astype(int))

                prev_frame = previous_img[iteration-1][0,:,:,:]
                intensity_range = np.max(prev_frame, axis=None) - np.min(prev_frame, axis=None)
                # save predicted next frames
                prev_frame = (prev_frame - np.min(prev_frame, axis=None)) * 255 / intensity_range
                cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_prev_1" + ".jpeg", prev_frame.astype(int))

                # Visualize middle flows

                # flowColor_2 = utils.flowToColor(flow_2[iteration-1][0,:,:,:].squeeze())
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_2" + ".jpeg", flowColor_2)
                # flowColor_3 = utils.flowToColor(flow_3[iteration-1][0,:,:,:].squeeze())
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_3" + ".jpeg", flowColor_3)
                # flowColor_4 = utils.flowToColor(flow_4[iteration-1][0,:,:,:].squeeze())
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_4" + ".jpeg", flowColor_4)
                # flowColor_5 = utils.flowToColor(flow_5[iteration-1][0,:,:,:].squeeze())
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_5" + ".jpeg", flowColor_5)
                # flowColor_6 = utils.flowToColor(flow_6[iteration-1][0,:,:,:].squeeze())
                # cv2.imwrite(FLAGS.train_log_dir + str(epoch) + "_" + str(iteration) + "_" + str(trainIter) + "_flowColor_6" + ".jpeg", flowColor_6)


            # print("Iteration %d/%d is Done" % (iteration, maxTestIter))

        # print np.concatenate(flow_p, axis=0).shape, np.concatenate(flow_gt, axis=0).shape
        # Calculate endpoint error
        f1 = np.concatenate(flow_1, axis=0)
        f2 = np.concatenate(flow_gt, axis=0)
        AEE = utils.flow_ee(f1, f2)
        # Calculate anguar error, maybe not necessary
        # AAE = utils.flow_ae(f1, f2)
        # print AEE, AAE
        # AE_tot, AAE = utils.flow_ae(np.concatenate(flow_p, axis=0), np.concatenate(flow_gt, axis=0))

        # calculate statistics
        if epoch == 1:
            print("***Test: max (flow_gt) %2.4f    abs_mean (flow_gt) %2.4f \r\n"
                % (np.amax(f2, axis=None), np.mean(np.absolute(f2), axis=None)))
        print("***Test flow abs_mean: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f pr6 %2.4f" 
            % (np.mean(np.absolute(flows_all[0]), axis=None), np.mean(np.absolute(flows_all[1]), axis=None), np.mean(np.absolute(flows_all[2]), axis=None), 
                np.mean(np.absolute(flows_all[3]), axis=None), np.mean(np.absolute(flows_all[4]), axis=None), np.mean(np.absolute(flows_all[5]), axis=None)))
        print("***Test flow max: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f pr6 %2.4f" 
            % (np.max(np.absolute(flows_all[0]), axis=None), np.max(np.absolute(flows_all[1]), axis=None), np.max(np.absolute(flows_all[2]), axis=None), 
                np.max(np.absolute(flows_all[3]), axis=None), np.max(np.absolute(flows_all[4]), axis=None), np.max(np.absolute(flows_all[5]), axis=None)))
        Loss_sum = (Loss1*weight_L[0] + Loss2*weight_L[1] + Loss3*weight_L[2] + Loss4*weight_L[3] + Loss5*weight_L[4] + Loss6*weight_L[5])/maxTestIter
        ULoss_sum = (U_Loss1*weight_L[0] + U_Loss2*weight_L[1] + U_Loss3*weight_L[2] + U_Loss4*weight_L[3] + U_Loss5*weight_L[4] + U_Loss6*weight_L[5])/maxTestIter*self.lambda_smooth
        VLoss_sum = (V_Loss1*weight_L[0] + V_Loss2*weight_L[1] + V_Loss3*weight_L[2] + V_Loss4*weight_L[3] + V_Loss5*weight_L[4] + V_Loss6*weight_L[5])/maxTestIter*self.lambda_smooth
        print("***Test: Epoch %03d Iter %04d: Loss_sum %4.4f ULoss_sum %4.4f VLoss_sum %4.4f AEE %4.4f \r\n" 
            % (epoch, trainIter, Loss_sum, ULoss_sum, VLoss_sum, AEE))