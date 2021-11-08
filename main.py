
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
   frameCount = 0
   frameExtractor(videopath,os.getcwd(),frameCount)

   img = cv2.imread(os.path.realpath('0000'+str(frameCount+1)+'.png'),cv2.IMREAD_UNCHANGED)
   print(featureExtractor.extract_feature(img))
   return featureExtractor.extract_feature(img)[0]


for oneFile in os.listdir(trainDataDir):
   try:

      print(getVectorFromVideo(os.path.join(trainDataDir,oneFile)))
      mapOfGestureToVector.append(getVectorFromVideo(os.path.join(trainDataDir,oneFile)))
   except:
      print("hello")

f = open("Results.csv", "w")
for testFile in sorted(os.listdir(testDataDir)):
   count = 0
   cosSimilarity = 0.0
   bestMatch = ""
   testVec = getVectorFromVideo(os.path.join(testDataDir, testFile))
   for trainVec in mapOfGestureToVector:
      cs = np.inner(testVec, trainVec)/(np.linalg.norm(testVec)*np.linalg.norm(trainVec))
      if cs > cosSimilarity:
         cosSimilarity = cs
         bestMatch = str(count)
         count = count + 1
   f.write(bestMatch+"\n")
f.close()