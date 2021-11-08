
import cv2
import os
def frameExtractor(videopath, frames_path,count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fbn= int(v/2)
    cap.set(1, fbn)
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(frames_path + "/%#05d.png" % (count+1), grayFrame)
