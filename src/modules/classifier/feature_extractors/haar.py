from modules.util.image_f import *
from featureHolder import FeatureHolder


class HaarFeature(object):
    def __init__(self, type, h, w, x, y):
        self.type = type
        self.h = int(h)
        self.w = int(w)
        self.x = int(x)
        self.y = int(y)
        # caching for speed
        self.hw = int(w/2)
        self.hh = int(h/2)
        self.tw = int(w/3)
        self.th = int(h/3)

    def __str__(self):
        return "Type: %d, height: %d, width: %d, x: %d, y: %d" % (self.type, self.h, self.w, self.x, self.y)

    def apply(self, img):  # TODO change signature to apply(self, patch)
        ret = None
        x = self.x
        y = self.y

        img = get_downsampled(img)

        if self.type == 0:  # horizontal
            #print 'A1= ' + str(x) + ',' + str(y)
            a1 = img[x, y]
            #print 'A2= ' + str(x + self.hw) + ',' + str(y)
            a2 = img[x + self.hw, y]
            #print 'B1= ' + str(x + self.hw - 1) + ',' + str(y)
            b1 = img[x + self.hw - 1, y]
            #print 'B2= ' + str(x + self.w - 1) + ',' + str(y)
            b2 = img[x + self.w - 1, y]
            #print 'C1= ' + str(x) + ',' + str(y + self.h - 1)
            c1 = img[x, y + self.h - 1]
            #print 'C2= ' + str(x + self.hw) + ',' + str(y + self.h - 1)
            c2 = img[x + self.hw, y + self.h - 1]
            #print 'D1= ' + str(x + self.hw - 1) + ',' + str(y + self.h - 1)
            d1 = img[x + self.hw - 1, y + self.h - 1]
            #print 'D2= ' + str(x + self.w - 1) + ',' + str(y + self.h - 1)
            d2 = img[x + self.w - 1, y + self.h - 1]
            rec1 = d1 + a1 - c1 - b1
            rec2 = d2 + a2 - c2 - b2
            ret = rec1 - rec2
        if self.type == 1:  # vertical
            #print 'D1= ' + str(x + self.w - 1) + ',' + str(y + self.hh - 1)
            d1 = img[x + self.w - 1, y + self.hh - 1]
            #print 'D2= ' + str(x + self.w - 1) + ',' + str(y + self.h - 1)
            d2 = img[x + self.w - 1, y + self.h - 1]
            #print 'A1= ' + str(x) + ',' + str(y)
            a1 = img[x, y]
            #print 'A2= ' + str(x) + ',' + str(y + self.hh)
            a2 = img[x, y + self.hh]
            #print 'B1= ' + str(x + self.w - 1) + ',' + str(y)
            b1 = img[x + self.w - 1, y]
            #print 'B2= ' + str(x + self.w - 1) + ',' + str(y + self.hh)
            b2 = img[x + self.w - 1, y + self.hh]
            #print 'C1= ' + str(x) + ',' + str(y + self.hh - 1)
            c1 = img[x, y + self.hh - 1]
            #print 'C2= ' + str(x) + ',' + str(y + self.h - 1)
            c2 = img[x, y + self.h - 1]
            rec1 = d1 + a1 - c1 - b1
            rec2 = d2 + a2 - c2 - b2
            ret = rec1 - rec2
        if self.type == 5:  # horizontal
            print 'D1= ' + str(x + self.tw - 1) + ',' + str(y + self.h - 1)
            d1 = img[x + self.tw - 1, y + self.h - 1]
            print 'D2= ' + str(x + 2 * self.tw - 1) + ',' + str(y + self.h - 1)
            d2 = img[x + 2 * self.tw - 1, y + self.h - 1]
            print 'D3= ' + str(x + 3 * self.tw - 1) + ',' + str(y + self.h - 1)
            d3 = img[x + 3 * self.tw - 1, y + self.h - 1]
            print 'A1= ' + str(x) + ',' + str(y)
            a1 = img[x, y]
            print 'A2= ' + str(x + self.tw) + ',' + str(y)
            a2 = img[x + self.tw, y]
            print 'A3= ' + str(x + 2 * self.tw) + ',' + str(y)
            a3 = img[x + 2 * self.tw, y]
            print 'B1= ' + str(x + self.tw - 1) + ',' + str(y)
            b1 = img[x + self.tw - 1, y]
            print 'B2= ' + str(x + 2 * self.tw - 1) + ',' + str(y)
            b2 = img[x + 2 * self.tw - 1, y]
            print 'B3= ' + str(x + 3 * self.tw - 1) + ',' + str(y)
            b3 = img[x + 3 * self.tw - 1, y]
            print 'C1= ' + str(x) + ',' + str(y + self.h - 1)
            c1 = img[x, y + self.h - 1]
            print 'C2= ' + str(x + self.tw) + ',' + str(y + self.h - 1)
            c2 = img[x + self.tw, y + self.h - 1]
            print 'C3= ' + str(x + 2 * self.tw) + ',' + str(y + self.h - 1)
            c3 = img[x + 2 * self.tw, y + self.h - 1]
            rec1 = d1 + a1 - c1 - b1
            rec2 = d2 + a2 - c2 - b2
            rec3 = d3 + a3 - c3 - b3
            ret = rec1 + rec3 - rec2
        if self.type == 3:  # vertical
            print 'D1= ' + str(x + self.w - 1) + ',' + str(y + self.th - 1)
            d1 = img[x + self.w - 1, y + self.th - 1]
            print 'D2= ' + str(x + self.w - 1) + ',' + str(y + 2*self.th - 1)
            d2 = img[x + self.w - 1, y + 2*self.th - 1]
            print 'D3= ' + str(x + self.w - 1) + ',' + str(y + 3*self.th - 1)
            d3 = img[x + self.w - 1, y + 3*self.th - 1]
            print 'A1= ' + str(x) + ',' + str(y)
            a1 = img[x,y]
            print 'A2= ' + str(x) + ',' + str(y + self.th)
            a2 = img[x, y + self.th]
            print 'A3= ' + str(x) + ',' + str(y + 2 * self.th)
            a3 = img[x, y + 2 * self.th]
            print 'B1= ' + str(x) + ',' + str(y + self.th - 1)
            b1 = img[x, y + self.th - 1]
            print 'B2= ' + str(x) + ',' + str(y + 2 * self.th - 1)
            b2 = img[x, y + 2 * self.th - 1]
            print 'B3= ' + str(x) + ',' + str(y + 3 * self.th - 1)
            b3 = img[x, y + 3 * self.th - 1]
            print 'C1= ' + str(x + self.w - 1) + ',' + str(y)
            c1 = img[x + self.w - 1, y]
            print 'C2= ' + str(x + self.w - 1) + ',' + str(y + self.th)
            c2 = img[x + self.w - 1, y + self.th]
            print 'C3= ' + str(x + self.w - 1) + ',' + str(y + 2 * self.th)
            c3 = img[x + self.w - 1, y + 2 * self.th]
            rec1 = d1 + a1 - c1 - b1
            rec2 = d2 + a2 - c2 - b2
            rec3 = d3 + a3 - c3 - b3
            ret = rec1 + rec3 - rec2
        if self.type == 2:  # four-rectangle features
            #print 'D1= ' + str(x + self.hw - 1) + ',' + str(y + self.hh - 1)
            d1 = img[x + self.hw - 1, y + self.hh - 1]
            #print 'D2= ' + str(x + self.w - 1) + ',' + str(y + self.hh - 1)
            d2 = img[x + self.w - 1, y + self.hh - 1]
            #print 'D3= ' + str(x + self.hw - 1) + ',' + str(y + self.h- 1)
            d3 = img[x + self.hw - 1, y + self.h- 1]
            #print 'D4= ' + str(x + self.w - 1) + ',' + str(y + self.h - 1)
            d4 = img[x + self.w - 1, y + self.h - 1]
            #print 'A1= ' + str(x) + ',' + str(y)
            a1 = img[x, y]
            #print 'A2= ' + str(x + self.hw) + ',' + str(y )
            a2 = img[x + self.hw, y]
            #print 'A3= ' + str(x) + ',' + str(y + self.hh)
            a3 = img[x, y + self.hh]
            #print 'A4= ' + str(x + self.hw) + ',' + str(y + self.hh)
            a4 = img[x + self.hw, y + self.hh]
            #print 'B1= ' + str(x + self.hw - 1) + ',' + str(y)
            b1 = img[x + self.hw - 1, y]
            #print 'B2= ' + str(x + self.w - 1) + ',' + str(y)
            b2 = img[x + self.w - 1, y]
            #print 'B3= ' + str(x + self.hw - 1) + ',' + str(y + self.hh)
            b3 = img[x + self.hw - 1, y + self.hh]
            #print 'B4= ' + str(x + self.w - 1) + ',' + str(y + self.hh)
            b4 = img[x + self.w - 1, y + self.hh]
            #print 'C1= ' + str(x) + ',' + str(y + self.hh - 1)
            c1 = img[x, y + self.hh - 1]
            #print 'C2= ' + str(x + self.hw) + ',' + str(y + self.hh - 1)
            c2 = img[x + self.hw, y + self.hh - 1]
            #print 'C3= ' + str(x) + ',' + str(y + self.h - 1)
            c3 = img[x, y + self.h - 1]
            #print 'C4= ' + str(x + self.hw) + ',' + str(y + self.h - 1)
            c4 = img[x + self.hw, y + self.h - 1]
            rec1 = d1 + a1 - b1 - c1
            rec2 = d2 + a2 - b2 - c2
            rec3 = d3 + a3 - b3 - c3
            rec4 = d4 + a4 - b4 - c4
            ret = rec2 + rec3 - rec1 - rec4
        return ret

    def __eq__(self, other):
        return self.h == other.h and self.w == other.h


class HaarHolder(FeatureHolder):

    def __init__(self, (h, w)):
        self.features = []
        features = np.array([[4, 8], [8, 4], [8, 8], [12, 4], [8, 8]])

        for feature in xrange(3):                                 # for each feature
            sizex = features[feature, 0]
            sizey = features[feature, 1]
            for x in range(w - sizex + 1):                        # for each pixel in width
                for y in range(h - sizey + 1):                    # for each pixel in height
                    for width in range(sizex, w - x + 1, sizex):  # for each width possible in window size
                        height = sizey/float(sizex) * width

                        # TODO
                        if height + y >= h:
                            continue
                        # edge features
                        feat = HaarFeature(feature, height, width, x, y)
                        self.features.append(feat)

    def get_features(self):
        return self.features

    def get(self):
        pass


# class HaarExtractor(object):
#
#     def update_patch_haar(self, patch):
#         features = self.extract_haar_features(patch.crop)
#         patch.update_haar_features(features)
#
#     @staticmethod
#     def extract_haar_features(crop):
#         """
#         Returns a dictionary containing <applied_feature, value> pairs
#         """
#
#             print "Features ", feature, ": ", time.time() - start_time, "seconds"
#             print "Size Haar: ", np.shape(ret[feat])
#         return ret

# if __name__ == '__main__':
#     hh = HaarHolder((24, 24))
#     for h in hh.features:
#         print h

    # im = np.random.rand(24, 24)
    # ii = get_integral_image(im)
    # h = HaarExtractor()
    # f = h.extract_haar_features(ii)
    # print f.keys()[0]