from abc import ABCMeta, abstractmethod

import cv2

class InputStream(object):

	@abstractmethod
	def getFrame(self):
		raise

	def showVideo(self):
		frameNo = 0
		ret, image = self.video.read()
		while ret:
			cv2.waitKey(30)
			cv2.imshow('Feed', image)
			ret, image = self.video.read()

			frameNo = frameNo + 1
			print frameNo
		cv2.destroyWindow("Feed")

class VideoFileIS(InputStream):

	def __init__(self, videoPath):
		self.videoPath = videoPath
		self.frameNum  = 0
		try:
			self.video = cv2.VideoCapture(videoPath)
		except:
			print "Could not open video file"
			raise


	def getFrame(self):
		ret, image = self.video.read()

		self.frameNum = self.frameNum + 1
		return image