from abc import abstractmethod
from modules.util.image_f import *
import numpy as np
import cv2


class InputStream(object):

    @abstractmethod
    def getRGBImage(self, frame):
        raise

    def show_video(self):
        frameNo = 0
        ret, image = self.video.read()
        while ret:
            cv2.waitKey(30)
            cv2.imshow('Feed', image)
            ret, image = self.video.read()

            frameNo += 1
            print frameNo
        cv2.destroyWindow("Feed")


class VideoFileIS(InputStream):

    def __init__(self, path):
        self.path = path
        self.frame_id = 0

        try:
            self.video = cv2.VideoCapture(path)
            self.length = self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            print "Opened %s training video file, containing %d frames" % (self.path, self.length)
        except:
            print "Could not open video file"
            raise

    def getRGBImage(self, frame):
        self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)
        ret, image = self.video.read()
        return rgb2gray(image)

    def get_grayscale_img(self, frame):
        self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)
        ret, image = self.video.read()
        image = np.array(image, dtype='int64')
        return rgb2gray(image)

    def getNextRGBImage(self):
        ret, image = self.video.read()
        self.frame_id += 1
        return rgb2gray(image)

    def imshow(self, frame, annotations):
        """
        Shows the image.py number: frame alongside with the annotations
        """
        self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)
        ret, image = self.video.read()

        a = str(len(annotations))
        cv2.imshow('inputStream.showImage(#' + str(frame) + ') Annotations: ' + a, image)

        for annotation in annotations:
            x1 = annotation.minX
            y1 = annotation.minY
            x2 = annotation.maxX
            y2 = annotation.maxY
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.waitKey()