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
            # print 'A1= ' + str(x) + ',' + str(y)
            # print 'A2= ' + str(x + self.w/2) + ',' + str(y)
            # print 'B1= ' + str(x + self.w/2 - 1) + ',' + str(y)
            # print 'B2= ' + str(x + self.w - 1) + ',' + str(y)
            # print 'C1= ' + str(x) + ',' + str(y + self.h - 1)
            # print 'C2= ' + str(x + self.w/2) + ',' + str(y + self.h - 1)
            # print 'D1= ' + str(x + self.hw/2 - 1) + ',' + str(y + self.h - 1)
            # print 'D2= ' + str(x + self.w - 1) + ',' + str(y + self.h - 1)
            # TODO: why always zero ?
            rec1 = img[x + self.hw - 1, y + self.h - 1] + img[x, y] - \
                   img[x + self.hw - 1, y] - img[x, y + self.h - 1]
            rec2 = img[x + self.w - 1, y + self.h - 1] + img[x + self.hw, y] - \
                   img[x + self.w - 1, y] - img[x + self.hw, y + self.h - 1]
            ret = rec1 - rec2
        if self.type == 1:  # vertical
            #print 'D1= ' + str(x + width - 1) + ',' + str(y + height / 2 - 1)
            #print 'D2= ' + str(x + width - 1) + ',' + str(y + height - 1)
            #print 'A1= ' + str(x) + ',' + str(y)
            #print 'A2= ' + str(x) + ',' + str(y + width)
            #print 'B1= ' + str(x + width - 1) + ',' + str(y)
            #print 'B2= ' + str(x + width - 1) + ',' + str(y + height / 2)
            #print 'C1= ' + str(x) + ',' + str(y + height / 2 - 1)
            #print 'C2= ' + str(x) + ',' + str(y + height - 1)
            rec1 = img[x + self.w - 1, y + self.hh - 1] + img[x, y] - img[x + self.w - 1, y] - img[x, y + self.hh - 1]
            rec2 = img[x + self.w - 1, y + self.h - 1] + img[x, y + self.h] - img[x + self.w - 1, y + self.hh] - img[x, y + self.h - 1]
            ret = rec1 - rec2
        if self.type == 2:  # horizontal
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
            rec1 = img[x + self.tw - 1, y + self.h - 1] + img[x, y] - img[x + self.tw - 1, y] - img[x,          y + self.h - 1]
            rec2 = img[x + 2*self.tw - 1, y + self.h - 1] + img[x + self.tw     , y] - img[x + 2*self.tw - 1, y] - img[x + self.tw,     y + self.h - 1]
            rec3 = img[x + self.w - 1, y + self.h - 1] + img[x + 2 * self.tw, y] - img[x + self.w - 1, y] - img[x + 2 * self.tw, y + self.h - 1]
            ret = rec1 + rec3 - rec2
        if self.type == 3:  # vertical
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
            rec1 = img[x + self.w - 1, y + self.th - 1] + img[x, y] - img[x, y + self.th - 1] - img[x + self.w - 1, y]
            rec2 = img[x + self.w - 1, y + 2 * self.th - 1] + img[x, y + self.th] - img[x, y + 2 * self.th - 1] - img[x + self.w - 1, y + self.th]
            rec3 = img[x + self.w - 1, y + self.h - 1] + img[x, y + 2 * self.th] - img[x, y + self.h - 1] - img[x + self.w - 1, y + 2 * self.th]
            ret = rec1 + rec3 - rec2
        if self.type == 4:  # four-rectangle features
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
            rec1 = img[x + self.hw - 1, y + self.hh-1] + img[x, y] - img[x + self.hw - 1, y] - img[x, y + self.hh - 1]
            rec2 = img[x + self.w - 1, y + self.hh-1] + img[x + self.hw, y] - img[x + self.w - 1, y] - img[x + self.hw, y + self.hh - 1]
            rec3 = img[x + self.hw - 1, y + self.h-1] + img[x, y + self.hh] - img[x + self.hw - 1, y + self.hh] - img[x, y + self.h - 1]
            rec4 = img[x + self.w - 1, y + self.h-1] + img[x + self.hw, y + self.hh] - img[x + self.w - 1, y + self.hh] - img[x + self.hw, y + self.h - 1]
            ret = rec2 + rec3 - rec1 - rec4
        return ret

    def __eq__(self, other):
        return self.h == other.h and self.w == other.h


class HaarHolder(FeatureHolder):

    def __init__(self, (h, w)):
        self.features = []
        features = np.array([[2, 4], [4, 2], [2, 6], [6, 2], [4, 4]])

        for feature in xrange(5):                                 # for each feature
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