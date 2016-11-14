import os,sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import cv2
import time
import warpflow
import utils as utils
from testOF import test

class train:
    '''
    Pipeline for training
    '''

    def __init__(self, opts):
        # Load the dataset
        if opts["dataset"] == "flyingChairs":
            from flyingChairsLoader import loader
        elif opts["dataset"] == "MPI-Sintel":
            from sintelLoader import loader
        elif opts["dataset"] == "ucf101":
            from ucf101Loader import loader
        else:
            print("No such dataset loader yet. ")
            sys.exit()

        self.loader = loader(opts)
        self.maxIterPerEpoch = int(np.floor(self.loader.trainNum/opts["batch_size"]))
        print("One epoch has %d iterations based on batch size %d." % (self.maxIterPerEpoch, opts["batch_size"]))

        self.display = 1    # Number of iterations to display training log
        self.test_interval = 1    # Number of iterations to evaluate the model performance
        self.maxEpochs = 110
        self.bilinearInit = True

        # Hyper params for computing loss
        self.mean = self.loader.mean    # Samples image mean in that dataset
        self.weight_L = [7,5,3,3,1]    # Loss weights schedule
        self.numLosses = len(self.weight_L) 

        lambda_smooth = 1.0
        epsilon = 0.0001 
        alpha_c = 0.25
        alpha_s = 0.37
        hyper_params = []
        hyper_params.append(lambda_smooth) 
        hyper_params.append(epsilon)  
        hyper_params.append(alpha_c)  
        hyper_params.append(alpha_s) 

        self.trainNet(opts, hyper_params)

    def load_deconv_weights(self, var, sess):
        f_shape = sess.run(var).shape
        width = f_shape[0]
        heigh = f_shape[0]
        f = np.ceil(width/2.0)
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

    def trainNet(self, opts, hyper_params):

        if not os.path.isdir(opts["log_dir"]):
            os.makedirs(opts["log_dir"], mode=0777)

        # Figure out the dimension information
        source, target, _ = self.loader.sampleTrain(opts["batch_size"])
        dimen_info = source.shape
        # print dimen_info

        with tf.device('/gpu:0'):
            source_imgs = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            target_imgs = tf.placeholder(tf.float32, [dimen_info[0], dimen_info[1], dimen_info[2], dimen_info[3]])
            sample_mean = tf.placeholder(tf.float32, [len(self.mean)])
            loss_weight = tf.placeholder(tf.float32, [self.numLosses])
            params = tf.placeholder(tf.float32, [len(hyper_params)])
            is_training = tf.placeholder(tf.bool, [])
            loss, midFlows, previous = warpflow.VGG16(source_imgs, 
                                                      target_imgs, 
                                                      sample_mean, 
                                                      loss_weight, 
                                                      params, 
                                                      is_training)
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
            print("Our network has %4.2fM number of parameters. " % (total_parameters/1000000.0))

            # Construct the train_op
            total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
            learning_rate = tf.placeholder(tf.float32, shape=[])
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

            # Build the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)
            # init = tf.initialize_all_variables()
            sess.run(tf.initialize_all_variables())
            
            # Use bilinear upsampling to initialize deconv filters
            bilinearInit = self.bilinearInit
            if bilinearInit:
                self.deconv_bilinearInit_vars = [var for var in model_vars if (var.name).startswith('up')] 
                for var in self.deconv_bilinearInit_vars:
                    if "weights" in var.name:
                        self.load_deconv_weights(var, sess)
                print("-----Done initializing deconv filters with bilinear upsampling------")
            
            # See if there is pre-trained network
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            ckpt = tf.train.get_checkpoint_state(opts["log_dir"])
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore from " +  ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            # Training starts...
            display = self.display     
            lr = opts["learning_rate"]

            for epoch in xrange(1, self.maxEpochs+1):
                print("Epoch %d    Learning Rate %f: \r\n" % (epoch, lr))
                epoch_start = time.time()
                for iteration in xrange(1, self.maxIterPerEpoch+1):
                    source, target, _ = self.loader.sampleTrain(opts["batch_size"])  

                    # Training
                    train_op.run(feed_dict = {source_imgs: source, 
                                              target_imgs: target, 
                                              sample_mean: self.mean, 
                                              loss_weight: self.weight_L,
                                              params: hyper_params,
                                              is_training: True, 
                                              learning_rate: lr}, session = sess)
                    
                    if iteration % display == 0:
                        losses, flows_all, loss_sum = sess.run([loss, midFlows, total_loss], 
                                 feed_dict = {source_imgs: source, 
                                              target_imgs: target,
                                              sample_mean: self.mean, 
                                              loss_weight: self.weight_L,
                                              params: hyper_params,
                                              is_training: False})

                        # Print training logs
                        assert len(losses) == self.numLosses, "We got different number of intermediate losses!"
                        print("Train Batch(%d): Epoch %03d Iter %04d: Loss_sum %4.4f \r\n" 
                            % (opts["batch_size"], epoch, iteration, loss_sum))
                        for loss_idx in xrange(self.numLosses):
                            print("    PhotometricLoss%d = %4.4f (* %2.4f = %2.4f loss)" 
                                % (loss_idx+1, losses[loss_idx]["Charbonnier_reconstruct"], self.weight_L[loss_idx], losses[loss_idx]["Charbonnier_reconstruct"] * self.weight_L[loss_idx]))
                        for loss_idx in xrange(self.numLosses):
                            print("    SmoothnessLossU%d = %4.4f (* %2.4f = %2.4f loss)" 
                                % (loss_idx+1, losses[loss_idx]["U_loss"], self.weight_L[loss_idx]*hyper_params[0], losses[loss_idx]["U_loss"] * self.weight_L[loss_idx]*hyper_params[0]))
                        for loss_idx in xrange(self.numLosses):
                            print("    SmoothnessLossV%d = %4.4f (* %2.4f = %2.4f loss)" 
                                % (loss_idx+1, losses[loss_idx]["V_loss"], self.weight_L[loss_idx]*hyper_params[0], losses[loss_idx]["V_loss"] * self.weight_L[loss_idx]*hyper_params[0]))
                        
                        assert not np.isnan(loss_sum).any(), 'Model diverged with loss = NaN'
                    
                    # Evaluation
                    if iteration == self.test_interval:   
                        print("Start evaluating ......")
                        test(opts, epoch, self.weight_L, hyper_params, sess, self.loader, 
                            loss, midFlows, previous, source_imgs, target_imgs, sample_mean, loss_weight, params, is_training)
                
                # Decay the learning rate
                if epoch % opts["num_epochs_per_decay"] == 0:
                    lr *= opts["lr_decay"]
                
                # Save the trained model snapshot
                if epoch % opts["save_interval_epoch"] == 0:
                    print("Save to " + opts["log_dir"] + str(epoch) + '_model.ckpt')
                    saver.save(sess, opts["log_dir"] + str(epoch) + '_model.ckpt')

                epoch_end = time.time()
                print("One epoch takes %2.2f minutes." % ((epoch_end - epoch_start)/60.0))

