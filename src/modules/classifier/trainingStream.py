import random
import cv2

from frameStream import VideoFileIS
from modules.classifier.annotationStream import AnnotationStream
from modules.datastructures.patch import Patch
from modules.util import image_f


class TrainingStream(object):

    def __init__(self, vid_path, annotation_path):
        self.input_stream = VideoFileIS(vid_path)
        self.annotation_stream = AnnotationStream(annotation_path)
        self.curr_image = None
        self.cursor = 0

    def extract_training_patches(self, l, negative_count_per_frame=10):
        ret = []
        while len(ret) < l+1:
            tmp = self.get_random_patches(negative_count_per_frame)
            ret.extend(tmp)
        return ret[0:l]

    def get_random_patches(self, negative_count=10):
        annos = self.annotation_stream.get_annotated_frame_ids()
        rand = random.randint(0, len(annos)-1)

        frame_id = annos[rand]

        annotations = self.annotation_stream.get_annotations(frame_id)
        img = self.input_stream.get_grayscale_img(frame_id)

        # self.imshow(frame_id)
        assert len(annotations) > 0
        return self._extract_patches(img, annotations, negative_count)

    def _extract_patches(self, img, annotations, negative_count):
        ret = self._extract_positive_patches(img, annotations)
        neg = self._extract_negative_patches(img, ret, negative_count)
        ret.extend(neg)
        return ret

    def _extract_positive_patches(self, img, annotations):
        ret = []
        for a in annotations:
            crop = img[a.ymin:a.ymax, a.xmin:a.xmax]
            p = Patch(crop, a.frame_id, (a.xmin, a.ymin, a.xmax, a.ymax), +1)
            ret.append(p)
        return ret

    def _extract_negative_patches(self, img, positives, negative_count):
        ret = []
        frame_id = positives[0].frame_id
        annotation_sizes = []
        for p in positives:
            annotation_sizes.append(p.size())

        while len(ret) < negative_count:
            random_size = annotation_sizes[random.randint(0, len(positives)) - 1]
            random_patch = image_f.get_random_patch(img, random_size, frame_id)
            if image_f.overlap(positives, random_patch):
                continue
            else:
                ret.append(random_patch)
        return ret

    def size(self):
        return len(self.annotation_stream.get_annotated_frame_ids())

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
            cv2.rectangle(self.curr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("trainingStream: showImage()", self.curr_image / 255)
        cv2.waitKey(5)