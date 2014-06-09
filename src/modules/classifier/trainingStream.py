import random
import numpy as np
import cv2

from frameStream import VideoFileIS
from modules.classifier.annotationStream import AnnotationStream
from modules.datastructures.patch import Patch


class TrainingStream(object):

    def __init__(self, vid_path, annotation_path):
        self.input_stream = VideoFileIS(vid_path)
        self.annotation_stream = AnnotationStream(annotation_path)
        self.curr_image = None
        self.cursor = 0

    def get_random_patches(self):
        annos = self.annotation_stream.get_annotated_frame_ids()
        rand = random.randint(0, len(annos)-1)

        frame_id = annos[rand]

        annotations = self.annotation_stream.get_annotations(frame_id)
        img = self.input_stream.get_grayscale_img(frame_id)

        self.imshow(frame_id)
        assert len(annotations) > 0
        return self.extract_patches(img, annotations)

    def extract_patches(self, img, annotations):
        ret = []
        for a in annotations:
            crop = img[a.ymin:a.ymax, a.xmin:a.xmax]
            p = Patch(crop, a)
            ret.append(p)
        return ret

    def get_t(self):
        pass

    def imshow(self, frame):
        print 'showing image #', frame
        annotations = self.annotation_stream.get_annotations(self.cursor)
        self.input_stream.imshow(frame, annotations)

    def show_current_image(self):
        print 'showing image #', self.cursor
        annotations = self.annotation_stream.get_annotations(self.cursor)

        for annotation in annotations:
            x1 = annotation.minX
            y1 = annotation.minY
            x2 = annotation.maxX
            y2 = annotation.maxY
            cv2.rectangle(self.curr_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("trainingStream: showImage()", self.curr_image / 255)
        cv2.waitKey(5)