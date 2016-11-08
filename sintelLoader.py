import os, sys
# from random import shuffle
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

    def __init__(self, data_path, image_size, split, time_step, passKey):
        self.data_path = data_path
        self.image_size = image_size
        self.img_path = os.path.join(self.data_path, 'training', passKey)
        self.time_step = time_step
        # self.split = split
        # Read in the standard train/val split from http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
        self.trainValGT = self.trainValSplit()
        self.trainList, self.valList = self.getMultiData(self.img_path, self.time_step)

        print("We have %d training samples and %d validation samples." % (len(self.trainList), len(self.valList)))
    
        if False:    # Calculate the global training data mean
            meanFile = self.calculateMean()
            print("B: %4.4f G: %4.4f R: %4.4f " % (meanFile[0], meanFile[1], meanFile[2]))
            # B: 70.1433 G: 83.1915 R: 92.8827 

        # assert(len(self.data) > 0)
        # print("Shuffling the data...")
        # shuffle(self.data)
        # splitCut = int(len(self.data)*split)
        # self.trainList = self.data[:splitCut]
        # self.valList = self.data[splitCut:]

    def trainValSplit(self):
        splitFile = "Sintel_train_val.txt"
        if not os.path.exists(splitFile):
            splitFileUrl = "http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/Sintel_train_val.txt"
            subprocess.call(["wget %s" % splitFileUrl], shell=True)

        with open(splitFile, 'r') as f:
            read_data = f.readlines()
            f.close()
            return read_data

    def getData(self, img_path):
        assert(os.path.exists(img_path))
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

    def getMultiData(self, img_path, time_step):
        assert(os.path.exists(img_path))
        clipDirs = os.listdir(img_path)
        clipDirs.sort()
        train = []
        val = []
        counter = 0
        for clip in clipDirs:
            clipDir = os.path.join(img_path, clip)
            imgDirs = os.listdir(clipDir)
            imgDirs.sort()
            for imgIdx in xrange(len(imgDirs)-time_step+1):
                frame_paths = []
                for idx in xrange(time_step):
                    frame_paths.append(os.path.join(clip, imgDirs[imgIdx + idx]))
                if self.trainValGT[counter][0] == "1":
                    train.append(frame_paths)
                elif self.trainValGT[counter][0] == "2":
                    val.append(frame_paths)
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
            multi_input = []
            multi_output = []
            multi_flow = []
            for time_idx in xrange(self.time_step-1):
                source = cv2.imread(os.path.join(self.img_path, img_pair[time_idx]), cv2.IMREAD_COLOR)
                target = cv2.imread(os.path.join(self.img_path, img_pair[time_idx+1]), cv2.IMREAD_COLOR)
                flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (img_pair[time_idx][:-4] + ".flo")))
                multi_input.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
                multi_output.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])), 0))
                # flow_gt.append(np.expand_dims(cv2.resize(flow, (self.image_size[1], self.image_size[0])), 0))
                multi_flow.append(np.expand_dims(flow, 0))
            source_list.append(np.concatenate(multi_input, axis=3))
            target_list.append(np.concatenate(multi_output, axis=3))
            flow_gt.append(np.concatenate(multi_flow, axis=3))
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
            img_pair = self.valList[idx]
            multi_input = []
            multi_output = []
            multi_flow = []
            for time_idx in xrange(self.time_step-1):
                source = cv2.imread(os.path.join(self.img_path, img_pair[time_idx]), cv2.IMREAD_COLOR)
                target = cv2.imread(os.path.join(self.img_path, img_pair[time_idx+1]), cv2.IMREAD_COLOR)
                flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (img_pair[time_idx][:-4] + ".flo")))
                multi_input.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
                multi_output.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])), 0))
                # flow_gt.append(np.expand_dims(cv2.resize(flow, (self.image_size[1], self.image_size[0])), 0))
                multi_flow.append(np.expand_dims(flow, 0))
            source_list.append(np.concatenate(multi_input, axis=3))
            target_list.append(np.concatenate(multi_output, axis=3))
            flow_gt.append(np.concatenate(multi_flow, axis=3))
            # prev_img = img_pair[0]
            # next_img = img_pair[1]
            # source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            # target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            # flow = utils.readFlow(os.path.join(self.data_path, 'training', "flow", (prev_img[:-4] + ".flo")))
            # source_list.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
            # target_list.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])) ,0))
            # # flow_gt.append(np.expand_dims(cv2.resize(flow, (self.image_size[1], self.image_size[0])), 0))
            # flow_gt.append(np.expand_dims(flow, 0))
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

            
