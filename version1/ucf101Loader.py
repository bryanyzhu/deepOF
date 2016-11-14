import os, sys
# from random import shuffle
import numpy as np
import cv2
import subprocess

class ucf101Loader:
    '''Pipeline for preparing the UCF101 data

    Image size: 240 x 320

    '''

    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size
        self.img_path = os.path.join(self.data_path, 'frames')
        # Read in the standard evaluation split 1
        self.trainDict, self.testDict, self.numClasses, self.trainLenClass, self.testLenClass = self.getData(self.img_path)
        print("There are %d action classes." % self.numClasses)
        total_train, total_test = 0, 0
        for cls in self.trainLenClass.keys():
            total_train += self.trainLenClass[cls]
            total_test += self.testLenClass[cls]
        print("We have %d training samples and %d validation samples." % (total_train, total_test))

    def getData(self, img_path):
        assert(os.path.exists(img_path))
        classDirs = os.listdir(img_path)
        classDirs.sort()
        classIndex = 0
        train = {}
        test = {}
        trainLenClass = {}
        testLenClass = {}
        for className in classDirs:
            classDir = os.path.join(img_path, className)
            clipDirs = os.listdir(classDir)
            clipDirs.sort()
            train[classIndex] = []
            test[classIndex] = []
            for clipName in clipDirs:
                if int(clipName.split("_")[2][1:]) > 7:
                    clipDir = os.path.join(classDir, clipName)
                    frameDirs = os.listdir(clipDir)
                    frameDirs.sort()
                    for frameIdx in xrange(len(frameDirs)-1):
                        prevFrame = os.path.join(className, clipName, frameDirs[frameIdx])
                        nextFrame = os.path.join(className, clipName, frameDirs[frameIdx+1])
                        train[classIndex].append((prevFrame, nextFrame))
                else:
                    clipDir = os.path.join(classDir, clipName)
                    frameDirs = os.listdir(clipDir)
                    frameDirs.sort()
                    for frameIdx in xrange(len(frameDirs)-1):
                        prevFrame = os.path.join(className, clipName, frameDirs[frameIdx])
                        nextFrame = os.path.join(className, clipName, frameDirs[frameIdx+1])
                        test[classIndex].append((prevFrame, nextFrame))
            print("Class %d: %s is attached." % (classIndex, className))
            trainLenClass[classIndex] = len(train[classIndex])
            testLenClass[classIndex] = len(test[classIndex])
            classIndex += 1
        print("Done preparing data.")
        return train, test, classIndex, trainLenClass, testLenClass

    def sampleTrain(self, batch_size):
        assert batch_size > 0, 'we need a batch size larger than 0'
        batchSampleIdxs = np.random.choice(self.numClasses, batch_size)
        # batchSampleIdxs = range((batch_id-1)*batch_size, batch_id*batch_size)
        return self.hookTrainData(batchSampleIdxs)

    def hookTrainData(self, sampleIdxs):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, label_list = [], [], []
        for idx in sampleIdxs:
            classList = self.trainDict[idx]
            img_pair = classList[np.random.choice(self.trainLenClass[idx], 1)]
            prev_img = img_pair[0]
            next_img = img_pair[1]
            label = idx
            # print prev_img, next_img, label
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            source_list.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
            target_list.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])) ,0))
            label_list.append(np.expand_dims(label, 0))
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(label_list, axis=0)
        # Adding the channel dimension if images are read in grayscale
        # return np.expand_dims(source_list, axis = 3), np.expand_dims(target_list, axis = 3)  

    def sampleVal(self, batch_size, classID):
        assert batch_size > 0, 'we need a batch size larger than 0'
        # batchSampleIdxs = np.random.choice(len(self.valList), batch_size)
        batchSampleIdxs = np.random.choice(self.testLenClass[classID], batch_size)
        # print batchSampleIdxs
        return self.hookValData(batchSampleIdxs, classID)

    def hookValData(self, sampleIdxs, classID):
        assert len(sampleIdxs) > 0, 'we need a non-empty batch list'
        source_list, target_list, label_list = [], [], []
        for idx in sampleIdxs:
            img_pair = self.testDict[classID][idx]
            prev_img = img_pair[0]
            next_img = img_pair[1]
            source = cv2.imread(os.path.join(self.img_path, prev_img), cv2.IMREAD_COLOR)
            target = cv2.imread(os.path.join(self.img_path, next_img), cv2.IMREAD_COLOR)
            source_list.append(np.expand_dims(cv2.resize(source, (self.image_size[1], self.image_size[0])), 0))
            target_list.append(np.expand_dims(cv2.resize(target, (self.image_size[1], self.image_size[0])) ,0))
            label_list.append(np.expand_dims(classID, 0))
            # print prev_img, next_img, classID
        return np.concatenate(source_list, axis=0), np.concatenate(target_list, axis=0), np.concatenate(label_list, axis=0)    

            

            
