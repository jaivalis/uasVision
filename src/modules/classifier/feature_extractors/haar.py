import numpy as np
import time


class HaarExtractor(object):

    def update_patch_haar(self, patch):
        features = self.extract_haar_features(patch.crop)
        patch.update_haar_features(features)

    @staticmethod
    def extract_haar_features(crop):
        w, h = crop.shape
        features = np.array([[2, 4], [4, 2], [2, 6], [6, 2], [4, 4]])
        Haar = []

        # for each feature
        for feature in xrange(5):
            start_time = time.time()
            sizex = features[feature, 0]
            sizey = features[feature, 1]
            # for each pixel in width
            for x in range(w - sizex + 1):
                # for each pixel in height
                for y in range(h - sizey + 1):
                    # for each width possible in window size
                    for width in range(sizex, w - x + 1, sizex):
                        height = sizey/float(sizex) * width

                        hw = width / 2
                        hh = height / 2
                        tw = width / 3
                        th = height / 3
                        # TODO
                        if height + y >= h:
                            continue
                            # edge features
                        if feature == 0:
                            # horizontal
                            #print 'D1= ' + str(x + width/2 - 1) + ',' + str(y + height - 1)
                            #print 'D2= ' + str(x + width - 1) + ',' + str(y + height - 1)
                            #print 'A1= ' + str(x) + ',' + str(y)
                            #print 'A2= ' + str(x + width/2) + ',' + str(y )
                            #print 'B1= ' + str(x + width/2 - 1) + ',' + str(y)
                            #print 'B2= ' + str(x + width - 1) + ',' + str(y)
                            #print 'C1= ' + str(x) + ',' + str(y + height - 1)
                            #print 'C2= ' + str(x + width/2) + ',' + str(y + height - 1)
                            rec1 = crop[x + hw - 1, y + height - 1] + crop[x        , y] - crop[x + hw - 1, y] - crop[x          , y + height - 1]
                            rec2 = crop[x + width - 1, y + height - 1] + crop[x + hw, y] - crop[x + width - 1, y] - crop[x + hw, y + height - 1]
                            Haar.append(rec1 - rec2)
                        elif feature == 1:
                            # vertical
                            #print 'D1= ' + str(x + width - 1) + ',' + str(y + height / 2 - 1)
                            #print 'D2= ' + str(x + width - 1) + ',' + str(y + height - 1)
                            #print 'A1= ' + str(x) + ',' + str(y)
                            #print 'A2= ' + str(x) + ',' + str(y + width)
                            #print 'B1= ' + str(x + width - 1) + ',' + str(y)
                            #print 'B2= ' + str(x + width - 1) + ',' + str(y + height / 2)
                            #print 'C1= ' + str(x) + ',' + str(y + height / 2 - 1)
                            #print 'C2= ' + str(x) + ',' + str(y + height - 1)
                            rec1 = crop[x + width - 1, y + hh - 1]     + crop[x, y]          - crop[x + width - 1, y]      - crop[x, y + hh - 1]
                            rec2 = crop[x + width - 1, y + height - 1] + crop[x, y + height] - crop[x + width - 1, y + hh] - crop[x, y + height - 1]
                            Haar.append(rec1 - rec2)
                            # line features
                        elif feature == 2:
                            # horizontal
                            #print 'D1= ' + str(x + width/3 - 1) + ',' + str(y + height - 1)
                            #print 'D2= ' + str(x + 2 * width/3 - 1) + ',' + str(y + height - 1)
                            #print 'D3= ' + str(x + 3 * width/3 - 1) + ',' + str(y + height - 1)
                            #print 'A1= ' + str(x) + ',' + str(y)
                            #print 'A2= ' + str(x + width/3) + ',' + str(y )
                            #print 'A3= ' + str(x + 2 * width/3) + ',' + str(y)
                            #print 'B1= ' + str(x + width/3 - 1) + ',' + str(y)
                            #print 'B2= ' + str(x + 2 * width/3 - 1) + ',' + str(y)
                            #print 'B3= ' + str(x + 3 * width/3 - 1) + ',' + str(y)
                            #print 'C1= ' + str(x) + ',' + str(y + height - 1)
                            #print 'C2= ' + str(x + width/3) + ',' + str(y + height - 1)
                            #print 'C3= ' + str(x + 2 * width/3) + ',' + str(y + height - 1)
                            rec1 = crop[x + tw - 1  , y + height - 1] + crop[x          , y] - crop[x + tw - 1  , y] - crop[x,          y + height - 1]
                            rec2 = crop[x + 2*tw - 1, y + height - 1] + crop[x + tw     , y] - crop[x + 2*tw - 1, y] - crop[x + tw,     y + height - 1]
                            rec3 = crop[x + width - 1, y + height - 1] + crop[x + 2 * tw, y] - crop[x + width - 1, y] - crop[x + 2 * tw,y + height - 1]
                            Haar.append(rec1 + rec3 - rec2)
                        elif feature == 3:
                            # vertical
                            #print 'D1= ' + str(x + width - 1) + ',' + str(y + height/3 - 1)
                            #print 'D2= ' + str(x + width - 1) + ',' + str(y + 2*height/3 - 1)
                            #print 'D3= ' + str(x + width - 1) + ',' + str(y + 3*height/3 - 1)
                            #print 'A1= ' + str(x) + ',' + str(y)
                            #print 'A2= ' + str(x) + ',' + str(y + height/3)
                            #print 'A3= ' + str(x) + ',' + str(y + 2 * height/3)
                            #print 'B1= ' + str(x) + ',' + str(y + height/3 - 1)
                            #print 'B2= ' + str(x) + ',' + str(y + 2 * height/3 - 1)
                            #print 'B3= ' + str(x) + ',' + str(y + 3 * height/3 - 1)
                            #print 'C1= ' + str(x + width - 1) + ',' + str(y)
                            #print 'C2= ' + str(x + width - 1) + ',' + str(y + height/3)
                            #print 'C3= ' + str(x + width - 1) + ',' + str(y + 2 * height/3)
                            rec1 = crop[x + width - 1, y + th - 1]     + crop[x, y]          - crop[x, y + th - 1]     - crop[x + width - 1, y]
                            rec2 = crop[x + width - 1, y + 2 * th - 1] + crop[x, y + th]     - crop[x, y + 2 * th - 1] - crop[x + width - 1, y + th]
                            rec3 = crop[x + width - 1, y + height - 1] + crop[x, y + 2 * th] - crop[x, y + height - 1] - crop[x + width - 1, y + 2 * th]
                            Haar.append(rec1 + rec3 - rec2)
                            # four-rectangle features
                        elif feature == 4:
                            #print 'D1= ' + str(x + width/2 - 1) + ',' + str(y + height/2 - 1)
                            #print 'D2= ' + str(x + width - 1) + ',' + str(y + height/2 - 1)
                            #print 'D3= ' + str(x + width/2 - 1) + ',' + str(y + height- 1)
                            #print 'D4= ' + str(x + width - 1) + ',' + str(y + height - 1)
                            #print 'A1= ' + str(x) + ',' + str(y)
                            #print 'A2= ' + str(x + width/2) + ',' + str(y )
                            #print 'A3= ' + str(x) + ',' + str(y + height/2)
                            #print 'A4= ' + str(x + width/2) + ',' + str(y + height/2)
                            #print 'B1= ' + str(x + width/2 - 1) + ',' + str(y)
                            #print 'B2= ' + str(x + width - 1) + ',' + str(y)
                            #print 'B3= ' + str(x + width/2 - 1) + ',' + str(y + height / 2)
                            #print 'B4= ' + str(x + width - 1) + ',' + str(y + height/2)
                            #print 'C1= ' + str(x) + ',' + str(y + height/2 - 1)
                            #print 'C2= ' + str(x + width/2) + ',' + str(y + height/2 - 1)
                            #print 'C3= ' + str(x) + ',' + str(y + height - 1)
                            #print 'C4= ' + str(x + width/2) + ',' + str(y + height - 1)
                            rec1 = crop[x + hw - 1, y + hh - 1]        + crop[x, y]           - crop[x + hw - 1, y]         - crop[x, y + hh - 1]
                            rec2 = crop[x + width - 1, y + hh - 1]     + crop[x + hw, y]      - crop[x + width - 1, y]      - crop[x + hw, y + hh - 1]
                            rec3 = crop[x + hw - 1, y + height- 1]     + crop[x, y + hh]      - crop[x + hw - 1, y + hh]    - crop[x, y + height - 1]
                            rec4 = crop[x + width - 1, y + height - 1] + crop[x + hw, y + hh] - crop[x + width - 1, y + hh] - crop[x + hw, y + height - 1]
                            Haar.append(rec2 + rec3 - rec1 - rec4)
            print "Features ", feature, ": ", time.time() - start_time, "seconds"
            print "Size Haar: ", np.shape(Haar)
        return Haar

if __name__ == '__main__':
    im = np.random.rand(24, 24)
    w, h = im.shape
    ii = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            ii[x, y] = np.sum(im[0:x+1, 0:y+1])
    h = HaarFeatureExtractor()
    f = h.extract_haar_features(ii)