import os, sys
import numpy as np
import cv2
import subprocess
import utils as utils

class sintelLoader:
    '''Pipeline for preparing the MPI Sintel data

    Image size: 1024 x 436
    Training images: 1064
    Training flows: 1041
    Training scenes: 23

    '''

    def __init__(self, data_path, image_size, time_step, passKey):
        self.data_path = data_path
        self.image_size = image_size
        self.img_path = os.path.join(self.data_path, 'training', passKey)
        self.time_step = time_step
        
        self.trainList = self.getTrainList(self.img_path)
        self.valList = self.getValList(self.img_path)
        print("We have %d training samples and %d validation samples." % (len(self.trainList), len(self.valList)))
        if False:    # Calculate the global training data mean
            meanFile = self.calculateMean()
            print("B: %4.4f G: %4.4f R: %4.4f " % (meanFile[0], meanFile[1], meanFile[2]))
            # B: 70.1433 G: 83.1915 R: 92.8827 

    def getTrainList(self, img_path):
        assert(os.path.exists(img_path))
        clipDirs = os.listdir(img_path)
        clipDirs.sort()
        train = []
        for clip in clipDirs:
            clipDir = os.path.join(img_path, clip)
            imgDirs = os.listdir(clipDir)
            imgDirs.sort()
            for imgIdx in xrange(len(imgDirs)-self.time_step+1):
                frame_paths = []
                for idx in xrange(self.time_step):
                    frame_paths.append(os.path.join(clip, imgDirs[imgIdx + idx]))
                train.append(frame_paths)
        return train

    def getValList(self, img_path):
        assert(os.path.exists(img_path))
        clipDirs = os.listdir(img_path)
        clipDirs.sort()
        val = []
        for clip in clipDirs:
            clipDir = os.path.join(img_path, clip)
            imgDirs = os.listdir(clipDir)
            imgDirs.sort()
            # Randomly select a start frame
            # imgIdx = np.random.randint(0, len(imgDirs)-self.time_step, 1)
            imgIdx = 0
            frame_paths = []
            for idx in xrange(self.time_step):
                frame_paths.append(os.path.join(clip, imgDirs[imgIdx + idx]))
            val.append(frame_paths)
            # In order to meet the batch size = 4, to make valList length 24, add one more sample
            if clip == "bamboo_2":      # We can change this to any folder
                # imgIdx = np.random.randint(0, len(imgDirs)-self.time_step, 1)
                frame_paths = []
                for idx in xrange(self.time_step):
                    frame_paths.append(os.path.join(clip, imgDirs[imgIdx + self.time_step + idx]))
                val.append(frame_paths)
        return val

    def sampleTrain(self, batch_size):
        assert batch_size > 0, 'we need a batch size larger than 0'
        batchSampleIdxs = np.random.choice(len(self.trainList), batch_size)
        return self.hookTrainData(batchSampleIdxs)

    def hookTrainData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        input_list, flow_list = [], []
        for idx in sampleIdxs:
            img_list = self.trainList[idx]
            multi_input = []
            multi_flow = []
            for time_idx in xrange(self.time_step):
                imgData = cv2.imread(os.path.join(self.img_path, img_list[time_idx]), cv2.IMREAD_COLOR)
                multi_input.append(np.expand_dims(cv2.resize(imgData, (self.image_size[1], self.image_size[0])), 0))
                # We have self.time_step images, but self.time_step - 1 flows.
                if time_idx != self.time_step - 1:
                    flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (img_list[time_idx][:-4] + ".flo")))
                    multi_flow.append(np.expand_dims(flow, 0))
            input_list.append(np.concatenate(multi_input, axis=3))
            flow_list.append(np.concatenate(multi_flow, axis=3))
        return np.concatenate(input_list, axis=0), np.concatenate(flow_list, axis=0)

    def sampleVal(self, batch_size, batch_id):
        assert batch_size > 0, 'we need a batch size larger than 0'
        # Do the evaluation in order, 24 samples, one batch of 4 samples
        batchSampleIdxs = xrange((batch_id-1)*batch_size, batch_id*batch_size)
        return (self.hookValData(batchSampleIdxs), batchSampleIdxs)

    def hookValData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        input_list, flow_list = [], []
        for idx in sampleIdxs:
            img_list = self.valList[idx]
            multi_input = []
            multi_flow = []
            for time_idx in xrange(self.time_step):
                imgData = cv2.imread(os.path.join(self.img_path, img_list[time_idx]), cv2.IMREAD_COLOR)
                multi_input.append(np.expand_dims(cv2.resize(imgData, (self.image_size[1], self.image_size[0])), 0))
                # We have self.time_step images, but self.time_step - 1 flows.
                if time_idx != self.time_step - 1:
                    flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (img_list[time_idx][:-4] + ".flo")))
                    multi_flow.append(np.expand_dims(flow, 0))
            input_list.append(np.concatenate(multi_input, axis=3))
            flow_list.append(np.concatenate(multi_flow, axis=3))
        return np.concatenate(input_list, axis=0), np.concatenate(flow_list, axis=0)     

    def calculateMean(self):
        numSamples = len(self.trainList)
        # OpenCV loads image as BGR order
        B, G, R = 0, 0, 0
        for idx in xrange(numSamples):
            frameID = self.trainList[idx]
            prev_img = frameID[0]
            next_img = frameID[1]
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

            
