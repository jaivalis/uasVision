import csv

from modules.datastructures.annotation import Annotation


class AnnotationStream(object):

    def __init__(self, path):
        detections_file = csv.reader(open(path, 'rb'), delimiter=' ')
        self.annotations = dict()

        for row in detections_file:
            ann = Annotation(-1, (row[1], row[2], row[3], row[4]), row[0], row[5], row[6], row[7])

            # Format: [trackID (minX minY maxX maxY) frameID lost occluded]
            if ann.lost == 1:  # discard empty frames
                continue
            if int(row[0]) not in self.annotations:
                self.annotations[int(row[0])] = list()
            self.annotations[int(row[0])].append(ann)

    def get_annotated_frame_ids(self):
        return self.annotations.keys()

    def get_annotations(self, frame_id):
        ret = self.annotations.get(frame_id)
        if ret is None:
            return []
        return ret