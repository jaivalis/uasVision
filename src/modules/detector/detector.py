import numpy as np
from modules.datastructures.patch import Patch
from scipy.misc import imresize
import cv2


class Detector(object):
    def __init__(self, classifier, frame_feed):
        self.classifier = classifier
        self.frame_feed = frame_feed

    def classify_all(self):
        frame = self.frame_feed.get_next_frame()
        print "Running detector for video file",
        dec = .05

        skip_rate = 10
        start_frame = 100

        while frame is not None:
            frame_num = self.frame_feed.frame_id
            if frame_num / skip_rate == 0 or frame_num < start_frame:
                frame = self.frame_feed.get_next_frame()
                continue
            # down sample frame to half its size
            orig_h, orig_w = np.shape(frame)
            new_h = orig_h / 2
            new_w = orig_w / 2
            frame = imresize(frame, (new_h, new_w))

            detections = self.detect(frame, self.frame_feed.frame_id)

            self.output_detections(frame, self.frame_feed.frame_id, detections)

            frame = self.frame_feed.get_next_frame()
            if self.frame_feed.frame_id > dec * len(self.frame_feed):
                dec += .05
                print ".",
        print "[DONE]"

    def detect(self, frame, frame_id):
        h, w = np.shape(frame)
        ret = []

        print h, w

        window_sizes = [(24, 24)]
        for y in range(0, h, 6):  # sliding window
            for x in range(0, w, 6):
                for window in window_sizes:
                    x_max = x + window[0]
                    y_max = y + window[1]
                    # print x, x_max, y, y_max
                    if x_max > w-1 or y_max > h-1:
                        continue
                    crop = frame[x:x_max, y:y_max]
                    p = Patch(crop, frame_id, (y, x, y_max, x_max), None)

                    label = self.classifier.classify(p)

                    if label == +1:
                        ret.append(p)
        return ret

    def output_detections(self, frame, frame_id, patches):
        """
        :param frame:
        :param patches: List containing the detections for a given frame
        """
        path = "../../annotated_frames/%05d.jpg" % frame_id
        for p in patches:
            # cv2.rectangle(frame, (p.xmin, p.xmax), (p.ymin, p.ymax), (255, 255, 255), 2)
            cv2.rectangle(frame, (p.xmin, p.ymin), (p.xmax, p.ymax), (255, 255, 255), 2)
        # cv2.imshow("annotated frame", frame)
        cv2.imwrite(path, frame)
        # cv2.waitKey()
        print "%s file written to disk" % path
