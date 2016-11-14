import os,sys
sys.path.append('./loader')
sys.path.append('./model')
sys.path.append('./utils')

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import time
import utils as utils

class test:
    '''
    Pipeline for evaluation
    '''

    def __init__(self, opts, epoch, weight_L, hyper_params, sess, loader,
                loss, midFlows, previous, source_imgs, target_imgs, sample_mean, loss_weight, params, is_training):
        self.loader = loader

        if opts["dataset"] == "flyingChairs":
            self.fc_test(opts, epoch, weight_L, hyper_params, sess, 
                loss, midFlows, previous, source_imgs, target_imgs, sample_mean, loss_weight, params, is_training)
        elif opts["dataset"] == "MPI-Sintel":
            self.sintel_test(opts, epoch, weight_L, hyper_params, sess, 
                loss, midFlows, previous, source_imgs, target_imgs, sample_mean, loss_weight, params, is_training)
        elif opts["dataset"] == "ucf101":
            self.ucf101_test(opts, epoch, weight_L, hyper_params, sess, 
                loss, midFlows, previous, source_imgs, target_imgs, sample_mean, loss_weight, params, is_training)
        else:
            print("No such dataset evaluation function yet. ")
            sys.exit()

    def fc_test(self, 
                opts,
                epoch,  
                weight_L,
                hyper_params, 
                sess, 
                loss, 
                midFlows, 
                previous,
                source_imgs, 
                target_imgs,
                sample_mean,
                loss_weight,
                params,
                is_training):

        testBatchSize = opts["batch_size"]
        maxTestIter = int(np.floor(self.loader.valNum/testBatchSize))
        Loss1, Loss2, Loss3, Loss4, Loss5 = 0,0,0,0,0
        U_Loss1, U_Loss2, U_Loss3, U_Loss4, U_Loss5 = 0,0,0,0,0
        V_Loss1, V_Loss2, V_Loss3, V_Loss4, V_Loss5 = 0,0,0,0,0
        flow_1 = []
        flow_gt = []
        previous_img = []
        for iteration in xrange(1, maxTestIter+1):
            testBatch = self.loader.sampleVal(testBatchSize, iteration)
            source, target, flow = testBatch[0]
            imgPath = testBatch[1][0]
            losses, flows_all, prev_all = sess.run([loss, midFlows, previous], 
                                    feed_dict = {source_imgs: source, 
                                                 target_imgs: target,
                                                 sample_mean: self.loader.mean, 
                                                 loss_weight: weight_L,
                                                 params: hyper_params,
                                                 is_training: False})
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
                flowImg = flows_all[0][batch_idx,:,:,:] * 2.0       # pr1 is still half of the final predicted flow value
                flowImg = np.clip(flowImg, -204.4790, 201.3478)       # 300 and 250 is the min and max of the flow value in training dataset
                # print flowImg.shape
                flow1_list.append(np.expand_dims(cv2.resize(flowImg, (512, 384)), 0))
                previous_img_list.append(np.expand_dims(cv2.resize(prev_all[batch_idx,:,:,:], (512, 384)), 0))
            flow_1.append(np.concatenate(flow1_list, axis=0))
            previous_img.append(np.concatenate(previous_img_list, axis=0))
            flow_gt.append(flow)

            # Visualize
            if iteration % 4 == 0:
                if epoch == 1:      # save ground truth images and flow
                    gt_1 = source[0, :, :, :].squeeze()
                    cv2.imwrite(opts["log_dir"] + self.loader.valList[imgPath] + "_img1.jpeg", gt_1)
                    gt_2 = target[0, :, :, :].squeeze()
                    cv2.imwrite(opts["log_dir"] + self.loader.valList[imgPath] + "_img2.jpeg", gt_2)
                    GTflowColor = utils.flowToColor(flow[0,:,:,:].squeeze())
                    cv2.imwrite(opts["log_dir"] + self.loader.valList[imgPath] + "_gt_flow.jpeg", GTflowColor)

                flowColor_1 = utils.flowToColor(flow_1[iteration-1][0,:,:,:].squeeze())
                # print flowColor.max(), flowColor.min(), flowColor.mean()
                cv2.imwrite(opts["log_dir"] + str(epoch) + "_" + str(iteration) + "_flowColor_1" + ".jpeg", flowColor_1)

                prev_frame = previous_img[iteration-1][0,:,:,:]
                intensity_range = np.max(prev_frame, axis=None) - np.min(prev_frame, axis=None)
                # save predicted next frames
                prev_frame = (prev_frame - np.min(prev_frame, axis=None)) * 255 / intensity_range
                cv2.imwrite(opts["log_dir"] + str(epoch) + "_" + str(iteration) + "_prev_1" + ".jpeg", prev_frame.astype(int))

        # Calculate endpoint error
        f1 = np.concatenate(flow_1, axis=0)
        f2 = np.concatenate(flow_gt, axis=0)
        AEE = utils.flow_ee(f1, f2)

        # Calculate statistics
        if epoch == 1:
            print("***Test: max (flow_gt) %2.4f  min (flow_gt) %2.4f  abs_mean (flow_gt) %2.4f \r\n"
                % (np.amax(f2, axis=None), np.amin(f2, axis=None), np.mean(np.absolute(f2), axis=None)))
        print("***Test flow max: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.max(flows_all[0], axis=None), np.max(flows_all[1], axis=None), np.max(flows_all[2], axis=None), 
                np.max(flows_all[3], axis=None), np.max(flows_all[4], axis=None)))
        print("***Test flow min: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.min(flows_all[0], axis=None), np.min(flows_all[1], axis=None), np.min(flows_all[2], axis=None), 
                np.min(flows_all[3], axis=None), np.min(flows_all[4], axis=None)))
        print("***Test flow abs_mean: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.mean(np.absolute(flows_all[0]), axis=None), np.mean(np.absolute(flows_all[1]), axis=None), np.mean(np.absolute(flows_all[2]), axis=None), 
                np.mean(np.absolute(flows_all[3]), axis=None), np.mean(np.absolute(flows_all[4]), axis=None)))

        Loss_sum = (Loss1*weight_L[0] + Loss2*weight_L[1] + Loss3*weight_L[2] + Loss4*weight_L[3] + Loss5*weight_L[4])/maxTestIter
        ULoss_sum = (U_Loss1*weight_L[0] + U_Loss2*weight_L[1] + U_Loss3*weight_L[2] + U_Loss4*weight_L[3] + U_Loss5*weight_L[4])/maxTestIter*hyper_params[0]
        VLoss_sum = (V_Loss1*weight_L[0] + V_Loss2*weight_L[1] + V_Loss3*weight_L[2] + V_Loss4*weight_L[3] + V_Loss5*weight_L[4])/maxTestIter*hyper_params[0]
        print("***Test: Epoch %03d : Loss_sum %4.4f ULoss_sum %4.4f VLoss_sum %4.4f AEE %4.4f \r\n" 
            % (epoch, Loss_sum, ULoss_sum, VLoss_sum, AEE))
        print("Evaluation done. \r\n")

    def sintel_test(self, 
                    opts,
                    epoch,  
                    weight_L,
                    hyper_params, 
                    sess, 
                    loss, 
                    midFlows, 
                    previous,
                    source_imgs, 
                    target_imgs,
                    sample_mean,
                    loss_weight,
                    params,
                    is_training):

        testBatchSize = opts["batch_size"]
        maxTestIter = int(np.floor(self.loader.valNum/testBatchSize))
        Loss1, Loss2, Loss3, Loss4, Loss5 = 0,0,0,0,0
        U_Loss1, U_Loss2, U_Loss3, U_Loss4, U_Loss5 = 0,0,0,0,0
        V_Loss1, V_Loss2, V_Loss3, V_Loss4, V_Loss5 = 0,0,0,0,0
        flow_1 = []
        flow_gt = []
        previous_img = []
        for iteration in xrange(1, maxTestIter+1):
            testBatch = self.loader.sampleVal(testBatchSize, iteration)
            source, target, flow = testBatch[0]
            imgPath = testBatch[1][0]
            losses, flows_all, prev_all = sess.run([loss, midFlows, previous], 
                                    feed_dict = {source_imgs: source, 
                                                 target_imgs: target,
                                                 sample_mean: self.loader.mean, 
                                                 loss_weight: weight_L,
                                                 params: hyper_params,
                                                 is_training: False})
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
                flowImg = flows_all[0][batch_idx,:,:,:]*100       # pr1 is still half of the final predicted flow value
                # flowImg = np.clip(flowImg, -248.968, 333.623)       # 300 and 250 is the min and max of the flow value in training dataset
                flow1_list.append(np.expand_dims(cv2.resize(flowImg, (1024, 436)), 0))
                previous_img_list.append(np.expand_dims(cv2.resize(prev_all[batch_idx,:,:,:], (1024, 436)), 0))
            flow_1.append(np.concatenate(flow1_list, axis=0))
            previous_img.append(np.concatenate(previous_img_list, axis=0))
            flow_gt.append(flow)

            # Visualize
            # if False:
            # if iteration % 10 == 0:
            if epoch == 1:      # save ground truth images and flow
                dirTuple = self.loader.valList[imgPath][0]
                dirSplit = dirTuple.split("/")
                dirName = dirSplit[0]
                frameName = dirSplit[1][0:10]
                imgName = dirName + "_" + frameName
                
                gt_1 = source[0, :, :, :].squeeze()
                cv2.imwrite(opts["log_dir"] + imgName + "_1.jpeg", gt_1)
                gt_2 = target[0, :, :, :].squeeze()
                cv2.imwrite(opts["log_dir"] + imgName + "_2.jpeg", gt_2)
                GTflowColor = utils.flowToColor(flow[0,:,:,:].squeeze())
                cv2.imwrite(opts["log_dir"] + imgName + "_gt_flow.jpeg", GTflowColor)

            flowColor_1 = utils.flowToColor(flow_1[iteration-1][0,:,:,:].squeeze())
            # print flowColor.max(), flowColor.min(), flowColor.mean()
            cv2.imwrite(opts["log_dir"] + str(epoch) + "_" + str(iteration) + "_flowColor_1" + ".jpeg", flowColor_1)

            prev_frame = previous_img[iteration-1][0,:,:,:]
            intensity_range = np.max(prev_frame, axis=None) - np.min(prev_frame, axis=None)
            # save predicted next frames
            prev_frame = (prev_frame - np.min(prev_frame, axis=None)) * 255 / intensity_range
            cv2.imwrite(opts["log_dir"] + str(epoch) + "_" + str(iteration) + "_prev_1" + ".jpeg", prev_frame.astype(int))

        # Calculate endpoint error
        f1 = np.concatenate(flow_1, axis=0)
        f2 = np.concatenate(flow_gt, axis=0)
        AEE = utils.flow_ee(f1, f2)

        # Calculate statistics
        if epoch == 1:
            print("***Test: max (flow_gt) %2.4f  min (flow_gt) %2.4f  abs_mean (flow_gt) %2.4f \r\n"
                % (np.amax(f2, axis=None), np.amin(f2, axis=None), np.mean(np.absolute(f2), axis=None)))
        print("***Test flow max: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.max(flows_all[0], axis=None), np.max(flows_all[1], axis=None), np.max(flows_all[2], axis=None), 
                np.max(flows_all[3], axis=None), np.max(flows_all[4], axis=None)))
        print("***Test flow min: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.min(flows_all[0], axis=None), np.min(flows_all[1], axis=None), np.min(flows_all[2], axis=None), 
                np.min(flows_all[3], axis=None), np.min(flows_all[4], axis=None)))
        print("***Test flow abs_mean: pr1 %2.4f pr2 %2.4f pr3 %2.4f pr4 %2.4f pr5 %2.4f " 
            % (np.mean(np.absolute(flows_all[0]), axis=None), np.mean(np.absolute(flows_all[1]), axis=None), np.mean(np.absolute(flows_all[2]), axis=None), 
                np.mean(np.absolute(flows_all[3]), axis=None), np.mean(np.absolute(flows_all[4]), axis=None)))

        Loss_sum = (Loss1*weight_L[0] + Loss2*weight_L[1] + Loss3*weight_L[2] + Loss4*weight_L[3] + Loss5*weight_L[4])/maxTestIter
        ULoss_sum = (U_Loss1*weight_L[0] + U_Loss2*weight_L[1] + U_Loss3*weight_L[2] + U_Loss4*weight_L[3] + U_Loss5*weight_L[4])/maxTestIter*hyper_params[0]
        VLoss_sum = (V_Loss1*weight_L[0] + V_Loss2*weight_L[1] + V_Loss3*weight_L[2] + V_Loss4*weight_L[3] + V_Loss5*weight_L[4])/maxTestIter*hyper_params[0]
        print("***Test: Epoch %03d : Loss_sum %4.4f ULoss_sum %4.4f VLoss_sum %4.4f AEE %4.4f \r\n" 
            % (epoch, Loss_sum, ULoss_sum, VLoss_sum, AEE))
        print("Evaluation done. \r\n")