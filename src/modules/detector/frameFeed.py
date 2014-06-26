from modules.util.image_f import *

import cv2


class FrameFeed(object):
    """
    Class used to extract frames from video
    """
    def __init__(self, path):
        self.path = path
        self.frame_id = 0
        self.video = None
        try:
            self.video = cv2.VideoCapture(path)
            self.length = self.video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            print "Opened %s video file, containing %d frames" % (self.path, self.length)
        except:
            print "Could not open video file"
            raise

    def get_next_frame(self):
        if not self.video.isOpened():
            return None
        ret, image = self.video.read()

        self.frame_id += 1
        return rgb2gray(image)