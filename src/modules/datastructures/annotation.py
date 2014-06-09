class Annotation(object):

    def __init__(self, track_id, (xmin, ymin, xmax, ymax), frame, lost, occld=None, genertd=None, label=None):
        self.track_id = int(track_id)
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.frame = int(frame)
        self.lost = int(lost)
        self.occld = int(occld)
        self.genertd = int(genertd)
        self.label = label

    def cstr(self):
        ret = str(self.frame) + ' ' + str(self.xmin) + ' ' + str(self.ymin) + ' '
        ret += str(self.xmax) + ' ' + str(self.ymax) + ' ' + str(self.lost) + ' '
        ret += str(self.occld) + ' ' + str(self.genertd)
        return ret

    def __gt__(self, other):
        return self.frame > other.frame

    def __eq__(self, other):
        return self.frame == other.frame

    def __str__(self):
        ret = 'TrackID: ' + str(self.track_id)
        ret += ' minX: ' + str(self.xmin)
        ret += ' minY: ' + str(self.ymin)
        ret += ' maxX: ' + str(self.xmax)
        ret += ' maxY: ' + str(self.ymax)
        ret += ' frame: ' + str(self.frame)
        ret += ' lost: ' + str(self.lost)
        ret += ' occld: ' + str(self.occld)
        ret += ' genertd: ' + str(self.genertd)
        ret += ' label: ' + str(self.label)
        return ret