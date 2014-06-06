import cv2

class Patch(object):

	def __init__(self, image, frameN, coords, height, width, label, haarFlag, hogFlag):
		self.image  = image
		self.frameN = frameN
		self.coords = coords
		self.h      = height
		self.w      = width
		self.label  = label
		
		self.haarF  = None
		self.hogF   = None

		if haarFlag:
			self.extractHaarFeatures()
		if hogFlag:
			self.extractHOGFeatures()

	def extractHaarFeatures(self):

		pass

	def extractHOGFeatures(self):
		
		pass

	def showImage(self):
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

