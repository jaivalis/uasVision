from modules.classifier.trainingStream import TrainingStream
from modules.classifier.feature_extractors.haar import HaarHolder
from modules.classifier.classifier import StrongClassifier

vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']
annPaths = ['../dataset/annotations/sanitized/COW809_1_sane.txt']

if __name__ == '__main__':
    negative_count = 10
    ts = TrainingStream(vidPaths[0], annPaths[0])

    # while len(ts.getNextTrainingPatches()) == 0:
    # 	print ts.frameNum
    # 	continue

    # while True:
        # ts.getNextTrainingPatches()
        # ts.showCurrentImage()
    patches = ts.get_random_patches(negative_count)

    # h = HaarExtractor()
    hh = HaarHolder((24, 24))

    alpha = .9
    beta = 0
    gamma = .8


    classifier = StrongClassifier(ts, hh, alpha, beta, gamma, 50)


    # for p in patches:
    #     # if p.label == -1:
    #     p.imshow()
    #     h.update_patch_haar(p)