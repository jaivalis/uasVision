import random
import cv2
import numpy as np

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

        self.unused_frames = self.annotation_stream.get_annotated_frame_ids()

    def extract_training_patches(self, l, negative_ratio=1.):
        ret = []
        print "Extracting", l, "training samples",
        dec = .05
        while len(ret) < l+1:
            tmp = self.get_random_patches(negative_ratio)
            ret.extend(tmp)
            if len(ret) > dec * (l+1):
                dec += .05
                print ".",
        print "[DONE]"
        return ret[0:l]

    def get_training_set(self, haar_holder, patch_count):
        training = np.zeros(shape=(patch_count, len(haar_holder)))
        labels = np.zeros(patch_count)

        patches = self.get_random_patches_no_replacement(patch_count, negative_ratio=1.)

        f_i = 0

        dec = .05
        print "Constructing training set consisting of %d samples" % (len(haar_holder)*patch_count),
        for feature in haar_holder.get_features():
            p_i = 0
            if f_i > len(haar_holder) * dec:
                print ".",
                dec += .05
            for patch in patches:
                response = feature.apply(patch.crop)
                training[p_i, f_i] = response
                labels[p_i] = patch.label

                p_i += 1
            f_i += 1
        print "[DONE]"

        return training, labels

    def get_minimal_training_set(self, haar_holder, patch_count):
        """ Returns a numpy array containing entries [feature_id, true_label, feature_response]
        :param haar_holder: Haar feature holder
        :param patch_count: Count of patches to be used
        :return: numpy ndarray of size(patch_count * len(haar_holder), 3)
        """
        ret_size = patch_count * len(haar_holder)
        ret = np.zeros(shape=(ret_size, 3))

        patches = self.get_random_patches_no_replacement(patch_count, negative_ratio=1.)

        ret_index = 0
        dec = .05
        print "Constructing training set consisting of %d samples" % ret_size,
        for feature in haar_holder.get_features():
            feature_id = feature.get_id()
            for patch in patches:
                if ret_index > ret_size * dec:
                    print ".",
                    dec += .05

                true_label = patch.label
                response = feature.apply(patch.crop)
                ret[ret_index, :] = [feature_id, true_label, response]
                ret_index += 1
        print "[DONE]"
        return ret

    def get_random_patches_no_replacement(self, training_set_size, negative_ratio=1.):
        ret = []

        while True:
            # without replacement bit
            frame_id = self.unused_frames[random.randint(0, len(self.unused_frames)-1)]
            self.unused_frames.remove(frame_id)

            annotations = self.annotation_stream.get_annotations(frame_id)
            img = self.input_stream.get_grayscale_img(frame_id)

            ret.extend(self._extract_patches(img, annotations, negative_ratio))

            if len(ret) > training_set_size:
                break
            if not self.unused_frames:
                print "No more frames to sample from, returning training set of size %d" % len(ret)
                break
        return ret[0:training_set_size]

    def get_random_patches(self, negative_ratio=1.):
        frame_ids = self.annotation_stream.get_annotated_frame_ids()
        rand = random.randint(0, len(frame_ids)-1)

        frame_id = frame_ids[rand]

        annotations = self.annotation_stream.get_annotations(frame_id)
        img = self.input_stream.get_grayscale_img(frame_id)

        # self.imshow(frame_id)
        return self._extract_patches(img, annotations, negative_ratio)

    def _extract_patches(self, img, annotations, negative_ratio):
        ret = self._extract_positive_patches(img, annotations)
        neg = self._extract_negative_patches(img, ret, len(ret) * negative_ratio)
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