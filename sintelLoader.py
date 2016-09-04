import os, sys
from random import shuffle
import numpy as np
import tensorflow as tf
import cv2

class sintelLoader:
    '''Pipeline for preparing the video data
    '''

    def __init__(self, data_path, image_size, split, passKey):
        self.data_path = data_path
        self.image_size = image_size
        self.img_path = os.path.join(self.data_path, 'training', passKey)
        self.split = split
        self.data = self.getData(self.img_path)
        assert(len(self.data) > 0)
        print("Shuffling the data...")
        shuffle(self.data)
        splitCut = int(len(self.data)*split)
        self.trainList = self.data[:splitCut]
        self.valList = self.data[splitCut:]
        print("We have %d training samples and %d validation samples." % (len(self.trainList), len(self.valList)))

    def getData(self, img_path):
        assert(os.path.exists(img_path))
        clipDirs = os.listdir(img_path)
        clipDirs.sort()
        data = []
        for clip in clipDirs:
            clipDir = os.path.join(img_path, clip)
            imgDirs = os.listdir(clipDir)
            imgDirs.sort()
            for imgIdx in xrange(len(imgDirs)-1):
                prevFrame = os.path.join(clip, imgDirs[imgIdx])
                nextFrame = os.path.join(clip, imgDirs[imgIdx+1])
                data.append((prevFrame, nextFrame))
        return data

    def sampleTrain(self, batch_size):
        assert batch_size > 0, 'we need a batch size larger than 0'
        batchSampleIdxs = np.random.choice(len(self.trainList), batch_size)
        return self.hookTrainData(batchSampleIdxs)

    def hookTrainData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list = [], []
        for idx in sampleIdxs:
            img_pair = self.trainList[idx]
            prev_img = img_pair[0]
            next_img = img_pair[1]
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            source_list.append(cv2.resize(source, (self.image_size[1], self.image_size[0])))
            target_list.append(cv2.resize(target, (self.image_size[1], self.image_size[0])))
        return source_list, target_list
        # Adding the channel dimension if images are read in grayscale
        # return np.expand_dims(source_list, axis = 3), np.expand_dims(target_list, axis = 3)          

            

            
