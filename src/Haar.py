import numpy as np
import time

def extract_haar_features(patch):
    w,h = patch.shape
    features = np.array( [ [ 2 , 4 ] , [ 4 , 2 ] , [ 2 , 6 ] , [ 6 , 2 ] , [ 4 , 4 ] ] )
    Haar = []
    
    # for each feature
    for feature in xrange(5):
        start_time = time.time()
        sizeX = features[ feature , 0 ]
        sizeY = features[ feature , 1 ]
        # for each pixel in width
        for x in range( w - sizeX + 1):
            # for each pixel in heigth
            for y in range( h - sizeY + 1 ):
                # for each width possible in window size
                for width in range( sizeX , w - x + 1 , sizeX ):
                    height = sizeY/float(sizeX) * width
                    
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
                        rec1 = patch[x + hw - 1, y + height - 1] + patch[x        , y] - patch[x + hw - 1, y] - patch[x          , y + height - 1]
                        rec2 = patch[x + width - 1, y + height - 1] + patch[x + hw, y] - patch[x + width - 1, y] - patch[x + hw, y + height - 1]
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
                        rec1 = patch[x + width - 1, y + hh - 1]     + patch[x, y]          - patch[x + width - 1, y]      - patch[x, y + hh - 1]
                        rec2 = patch[x + width - 1, y + height - 1] + patch[x, y + height] - patch[x + width - 1, y + hh] - patch[x, y + height - 1]                            
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
                        rec1 = patch[x + tw - 1  , y + height - 1] + patch[x          , y] - patch[x + tw - 1  , y] - patch[x,          y + height - 1]
                        rec2 = patch[x + 2*tw - 1, y + height - 1] + patch[x + tw     , y] - patch[x + 2*tw - 1, y] - patch[x + tw,     y + height - 1]
                        rec3 = patch[x + width - 1, y + height - 1] + patch[x + 2 * tw, y] - patch[x + width - 1, y] - patch[x + 2 * tw,y + height - 1]
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
                        rec1 = patch[x + width - 1, y + th - 1]     + patch[x, y]          - patch[x, y + th - 1]     - patch[x + width - 1, y]
                        rec2 = patch[x + width - 1, y + 2 * th - 1] + patch[x, y + th]     - patch[x, y + 2 * th - 1] - patch[x + width - 1, y + th]                            
                        rec3 = patch[x + width - 1, y + height - 1] + patch[x, y + 2 * th] - patch[x, y + height - 1] - patch[x + width - 1, y + 2 * th]                            
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
                        rec1 = patch[x + hw - 1, y + hh - 1]        + patch[x, y]           - patch[x + hw - 1, y]         - patch[x, y + hh - 1]
                        rec2 = patch[x + width - 1, y + hh - 1]     + patch[x + hw, y]      - patch[x + width - 1, y]      - patch[x + hw, y + hh - 1]
                        rec3 = patch[x + hw - 1, y + height- 1]     + patch[x, y + hh]      - patch[x + hw - 1, y + hh]    - patch[x, y + height - 1]                            
                        rec4 = patch[x + width - 1, y + height - 1] + patch[x + hw, y + hh] - patch[x + width - 1, y + hh] - patch[x + hw, y + height - 1]
                        Haar.append(rec2 + rec3 - rec1 - rec4)
        print "Features ", feature, ": ", time.time() - start_time, "seconds"
        print "Size Haar: ", shape(Haar)
    return Haar
    
    
if __name__ == '__main__': 
    im = np.random.rand(24,24)
    w,h = im.shape
    ii = np.zeros((w,h))
    for x in range(w):
        for y in range(h):
            ii[x,y] = sum(im[0:x+1,0:y+1])
    Haar = extract_haar_features(ii)