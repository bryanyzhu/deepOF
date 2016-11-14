import os, sys
sys.path.append('./utils')
import numpy as np
import cv2
import subprocess
import utils as utils

class loader:
    '''Pipeline for preparing the MPI Sintel data

    Image size: 1024 x 436
    Training images: 1064
    Training flows: 1041
    Training scenes: 23

    '''

    def __init__(self, opts):
        self.data_path = opts["data_path"]
        passKey = "final"
        self.img_path = os.path.join(self.data_path, 'training', passKey)
        self.crop_size = [320, 512]    # TODO: aspect ratio is seriously changed.
        self.is_crop = opts["is_crop"]
        # Read in the standard train/val split from http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
        self.trainValGT = self.trainValSplit()
        self.trainList, self.valList = self.getData(self.img_path, opts["dataset"])
        self.trainNum = len(self.trainList)
        self.valNum = len(self.valList)
        print("We have %d training samples and %d validation samples." % (self.trainNum, self.valNum))
    
        if False:    # Calculate the global training data mean
            meanFile = self.calculateMean()
            print("B: %4.4f G: %4.4f R: %4.4f " % (meanFile[0], meanFile[1], meanFile[2]))
            # B: 70.1433 G: 83.1915 R: 92.8827 
        self.mean = np.array([70.1433, 83.1915, 92.8827], dtype=np.float32)

    def trainValSplit(self):
        splitFile = "./loader/Sintel_train_val.txt"
        if not os.path.exists(splitFile):
            splitFileUrl = "http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/Sintel_train_val.txt"
            subprocess.call(["wget %s -P ./loader/" % splitFileUrl], shell=True)

        with open(splitFile, 'r') as f:
            read_data = f.readlines()
            f.close()
            return read_data

    def getData(self, img_path, dataset):
        assert(os.path.exists(img_path))
        numSamples = len(self.trainValGT)
        print("There are %d image pairs in the %s dataset. " % (numSamples, dataset))
        clipDirs = os.listdir(img_path)
        clipDirs.sort()
        train = []
        val = []
        counter = 0
        for clip in clipDirs:
            clipDir = os.path.join(img_path, clip)
            imgDirs = os.listdir(clipDir)
            imgDirs.sort()
            for imgIdx in xrange(len(imgDirs)-1):
                prevFrame = os.path.join(clip, imgDirs[imgIdx])
                nextFrame = os.path.join(clip, imgDirs[imgIdx+1])
                if self.trainValGT[counter][0] == "1":
                    train.append((prevFrame, nextFrame))
                elif self.trainValGT[counter][0] == "2":
                    val.append((prevFrame, nextFrame))
                else:
                    print("Something wrong with the split file.")
                counter += 1
        return train, val

    def sampleTrain(self, batch_size):
        assert batch_size > 0, 'we need a batch size larger than 0'
        batchSampleIdxs = np.random.choice(len(self.trainList), batch_size)
        # batchSampleIdxs = range((batch_id-1)*batch_size, batch_id*batch_size)
        return self.hookTrainData(batchSampleIdxs)

    def hookTrainData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, flow_gt = [], [], []
        for idx in sampleIdxs:
            img_pair = self.trainList[idx]
            prev_img = img_pair[0]
            next_img = img_pair[1]
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (prev_img[:-4] + ".flo")))
            if self.is_crop:
                source = cv2.resize(source, (self.crop_size[1], self.crop_size[0]))
                target = cv2.resize(target, (self.crop_size[1], self.crop_size[0]))
            source_list.append(np.expand_dims(source, 0))
            target_list.append(np.expand_dims(target, 0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)

    def sampleVal(self, batch_size, batch_id):
        assert batch_size > 0, 'we need a batch size larger than 0'
        # batchSampleIdxs = np.random.choice(len(self.valList), batch_size)
        batchSampleIdxs = range((batch_id-1)*batch_size, batch_id*batch_size)
        return (self.hookValData(batchSampleIdxs), batchSampleIdxs)

    def hookValData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, flow_gt = [], [], []
        for idx in sampleIdxs:
            img_pair = self.valList[idx]
            prev_img = img_pair[0]
            next_img = img_pair[1]
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (prev_img[:-4] + ".flo")))
            if self.is_crop:
                source = cv2.resize(source, (self.crop_size[1], self.crop_size[0]))
                target = cv2.resize(target, (self.crop_size[1], self.crop_size[0]))
            source_list.append(np.expand_dims(source, 0))
            target_list.append(np.expand_dims(target, 0))
            flow_gt.append(np.expand_dims(flow, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(flow_gt, axis=0)       

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

            
