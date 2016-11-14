import os, sys
import numpy as np
import cv2
import subprocess
import utils as utils
import random

class loader:
    '''Pipeline for preparing the Flying Chairs data

    Image size: 512 x 384
    All image pairs: 22872
    Train: 22232
    Test: 640

    '''

    def __init__(self, opts):
        self.data_path = opts["data_path"]
        self.img_path = os.path.join(self.data_path, 'data')
        # Read in the standard train/val split from http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
        self.trainValGT = self.trainValSplit()
        self.trainList, self.valList = self.getData(self.img_path, opts["dataset"])
        self.trainNum = len(self.trainList)
        self.valNum = len(self.valList)
        print("We have %d training samples and %d validation samples." % (self.trainNum, self.valNum))
        # # Shuffle the training list 
        # print("Shuffling the training data.")
        # random.shuffle(self.trainList)
        if False:    # Calculate the global training data mean
            meanFile = self.calculateMean()
            print("B: %4.4f G: %4.4f R: %4.4f " % (meanFile[0], meanFile[1], meanFile[2]))
            # [97.533268117955444, 99.238235788550085, 97.055973199626948]
        self.mean = np.array([97.533268117955444, 99.238235788550085, 97.055973199626948], dtype=np.float32)
            
    def trainValSplit(self):
        splitFile = "FlyingChairs_train_val.txt"
        if not os.path.exists(splitFile):
            splitFileUrl = "http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt"
            subprocess.call(["wget %s" % splitFileUrl], shell=True)

        with open(splitFile, 'r') as f:
            read_data = f.readlines()
            f.close()
            return read_data

    def getData(self, img_path, dataset):
        assert(os.path.exists(img_path))
        numSamples = len(self.trainValGT)
        print("There are %d image pairs in the %s dataset. " % (numSamples, dataset))
        train = []
        val = []
        for imgIdx in xrange(numSamples):
            frameID = "%05d" % (imgIdx + 1)     # The image index starts at 00001
            if self.trainValGT[imgIdx][0] == "1":
                train.append(frameID)
            elif self.trainValGT[imgIdx][0] == "2":
                val.append(frameID)
            else:
                print("Something wrong with the split file.")
        return train, val

    # def sampleTrain(self, batch_size, batch_id):        # This will cover the whole dataset
    def sampleTrain(self, batch_size):
        assert batch_size > 0, 'we need a batch size larger than 0'
        batchSampleIdxs = np.random.choice(self.trainNum, batch_size)
        # batchSampleIdxs = xrange((batch_id-1)*batch_size, batch_id*batch_size)
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
            # No need to crop the image
            # source = cv2.resize(source, (self.image_size[1], self.image_size[0]))
            source_list.append(np.expand_dims(source, 0))
            # target = cv2.resize(target, (self.image_size[1], self.image_size[0]))
            target_list.append(np.expand_dims(target ,0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)

    def sampleVal(self, batch_size, batch_id):
        assert batch_size > 0, 'we need a batch size larger than 0'
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
            source_list.append(np.expand_dims(source, 0))
            target_list.append(np.expand_dims(target ,0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)     

    def calculateMean(self):
        numSamples = self.trainNum
        # OpenCV loads image as BGR order
        B, G, R = 0, 0, 0
        for idx in xrange(numSamples):
            frameID = self.trainList[idx]
            prev_img = frameID + "_img1.ppm"
            next_img = frameID + "_img2.ppm"
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            B += np.mean(source[:,:,0], axis=None)
            B += np.mean(target[:,:,0], axis=None)
            G += np.mean(source[:,:,1], axis=None)
            G += np.mean(target[:,:,1], axis=None)
            R += np.mean(source[:,:,2], axis=None)
            R += np.mean(target[:,:,2], axis=None)
        B = B / (2*numSamples)
        G = G / (2*numSamples)
        R = R / (2*numSamples)
        return (B,G,R)


            

            