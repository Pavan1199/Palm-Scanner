
import cv2
import numpy as np
import os
import tensorflow as tf

import math
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
featureExtractor = HandShapeFeatureExtractor()
trainDataDir = os.path.realpath('traindata')
testDataDir = os.path.realpath('test')
mapOfGestureToVector = []


def getVectorFromVideo(videopath):
   f = 0
   frameExtractor(videopath,os.getcwd(),f)

   i = cv2.imread(os.path.realpath('0000'+str(f+1)+'.png'),cv2.IMREAD_UNCHANGED)
   print(featureExtractor.extract_feature(i))
   return featureExtractor.extract_feature(i)[0]


for oneFile in os.listdir(trainDataDir):
   try:

      print(getVectorFromVideo(os.path.join(trainDataDir,oneFile)))
      mapOfGestureToVector.append(getVectorFromVideo(os.path.join(trainDataDir,oneFile)))
   except:
      print("hello")

f = open("Results.csv", "w")
for testFile in sorted(os.listdir(testDataDir)):
   c = 0
   cc = 0.0
   bm = ""
   tv = getVectorFromVideo(os.path.join(testDataDir, testFile))
   for trainVec in mapOfGestureToVector:
      cs = np.inner(tv, trainVec)/(np.linalg.norm(tv)*np.linalg.norm(trainVec))
      if cs > cc:
         cc = cs
         bm = str(c)
         c = c + 1
   f.write(bm+"\n")
f.close()