from os import listdir

from modules.datastructures.annotation import Annotation


def save_sane_file(path, saneList):
    with open(path, "w") as f:
        for sane in saneList:
            f.write( sane.cstr() + '\n' )


def parse_file(path):
    sane = []
    with open(path, "r") as f:
        for line in f:
            fields = line.split(' ')

            annot = Annotation(fields[0], (fields[1], fields[2], fields[3], fields[4]),
                                fields[5], fields[6], fields[7], fields[8], fields[9])
            sane.append(annot)
    sane.sort()
    return sane


if __name__ == '__main__':
    annotationsPath = '../dataset/annotations/'
    for f in listdir(annotationsPath):
        if f.endswith('.txt'):
            sanePath = annotationsPath + 'sanitized/' + f[0:-4] + '_sane.txt'
            print sanePath
            sane = parse_file(annotationsPath + f)

            save_sane_file(sanePath, sane)