import numpy as np
import cv2

vidPath = "../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4"

try:
	video = cv2.VideoCapture(vidPath)
except:
	print "Could not open video file"
	raise


ret, image = video.read()
print ret

frameNo = 0;

cv2.namedWindow("Feed")
while ret:
	cv2.waitKey(30)
	cv2.imshow('Feed', image)
	ret, image = video.read()

	frameNo = frameNo + 1
	print frameNo
    
cv2.destroyWindow("Feed")