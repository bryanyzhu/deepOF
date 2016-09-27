import os, sys
# from random import shuffle
import numpy as np
import cv2
import subprocess
import utils as utils

class flyingChairsLoader:
    '''Pipeline for preparing the Flying Chairs data

    Image size: 1024 x 436
    Training image pairs: 22872

    '''

    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size
        self.img_path = os.path.join(self.data_path, 'data')
        # self.split = split
        # Read in the standard train/val split from http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
        self.trainValGT = self.trainValSplit()
        self.trainList, self.valList = self.getData(self.img_path)
        print("We have %d training samples and %d validation samples." % (len(self.trainList), len(self.valList)))

    def trainValSplit(self):
        splitFile = "FlyingChairs_train_val.txt"
        if not os.path.exists(splitFile):
            splitFileUrl = "http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt"
            subprocess.call(["wget %s" % splitFileUrl], shell=True)

        with open(splitFile, 'r') as f:
            read_data = f.readlines()
            f.close()
            return read_data

    def getData(self, img_path):
        assert(os.path.exists(img_path))
        numSamples = len(self.trainValGT)
        print numSamples
        train = []
        val = []
        for imgIdx in xrange(numSamples):
            frameID = "%05d" % (imgIdx + 1)
            if self.trainValGT[imgIdx][0] == "1":
                train.append(frameID)
            elif self.trainValGT[imgIdx][0] == "2":
                val.append(frameID)
            else:
                print("Something wrong with the split file.")
        return train, val

    def sampleTrain(self, batch_size, batch_id):
        assert batch_size > 0, 'we need a batch size larger than 0'
        # batchSampleIdxs = np.random.choice(len(self.trainList), batch_size)
        batchSampleIdxs = xrange((batch_id-1)*batch_size, batch_id*batch_size)
        return self.hookTrainData(batchSampleIdxs)

    def hookTrainData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, flow_gt = [], [], []
        for idx in sampleIdxs:
            frameID = self.trainList[idx]
            prev_img = frameID + "_img1.ppm"
            next_img = frameID + "_img2.ppm"
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            # print source.shape
            flow = utils.readFlow(os.path.join(self.img_path, (frameID + "_flow.flo")))
            # print flow.shape
            source_list.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
            target_list.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])) ,0))
            # flow_gt.append(np.expand_dims(cv2.resize(flow, (self.image_size[1], self.image_size[0])), 0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)
        # Adding the channel dimension if images are read in grayscale
        # return np.expand_dims(source_list, axis = 3), np.expand_dims(target_list, axis = 3)  

    def sampleVal(self, batch_size, batch_id):
        assert batch_size > 0, 'we need a batch size larger than 0'
        # batchSampleIdxs = np.random.choice(len(self.valList), batch_size)
        batchSampleIdxs = range((batch_id-1)*batch_size, batch_id*batch_size)
        return (self.hookValData(batchSampleIdxs), batchSampleIdxs)

    def hookValData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, flow_gt = [], [], []
        for idx in sampleIdxs:
            frameID = self.valList[idx]
            prev_img = frameID + "_img1.ppm"
            next_img = frameID + "_img2.ppm"
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            flow = utils.readFlow(os.path.join(self.img_path, (frameID + "_flow.flo")))
            source_list.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
            target_list.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])) ,0))
            # flow_gt.append(np.expand_dims(cv2.resize(flow, (self.image_size[1], self.image_size[0])), 0))
            # flow_gt.append(np.expand_dims(flow, 0))e(flow, (self.image_size[1], self.image_size[0])), 0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)       

            

            
