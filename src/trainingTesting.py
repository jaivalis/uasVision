from modules.classifier.trainingStream import TrainingStream

vidPaths = ['../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4']
annPaths = ['../dataset/annotations/sanitized/COW809_1_sane.txt']

if __name__ == '__main__':
    ts = TrainingStream(vidPaths[0], annPaths[0])

    # while len(ts.getNextTrainingPatches()) == 0:
    # 	print ts.frameNum
    # 	continue

    # while True:
        # ts.getNextTrainingPatches()
        # ts.showCurrentImage()
    patches = ts.get_random_patches()

    for p in patches:
        p.imshow()