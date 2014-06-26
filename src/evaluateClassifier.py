from modules.detector.frameFeed import FrameFeed
from modules.detector.detector import Detector
from modules.util.serializer import *

vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']


if __name__ == '__main__':
    print os.getcwd()

    pickles = get_pickles(path='../../classifiers')
    if len(pickles) == 0:
        print "No saved classifiers found. Train one and output it to file"
    else:
        classifier = deserialize(pickles[-1])
        frame_feed = FrameFeed(vidPaths[-1])

        detector = Detector(classifier, frame_feed)

        detector.classify_all()
