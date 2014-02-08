"""
Image Processing for GalaxyZoo
author: sihrc
2/7/2014
"""
import os, cv2, cv
import numpy as np
import pickle as p
import time

class _Developing(object):
	"""
	Template Class for Developing Classes
	"""
	def save(self):
		"""
		Saves the Image Object for use in next session
		"""
		with open("temp.p", 'wb') as f:
			p.dump(self, f)
		self.log(msg = "Saved Successfully")

	def load(self):
		"""
		Loads Image Object from the previous session
		"""
		if not os.path.exists("temp.p"):
			self.save()
		with open("temp.p", 'rb') as f:
			self = p.load(f)
		self.log(msg = "Loaded Successfully")

	def log(self, msg = "Here" ,value = ""):
		identifier = "<%s>%s" % (self.__class__.__name__, self.name)
		print "DEBUGGER:",identifier, "\n"+70*"="+"\n", msg,"\n",value, "\n"


class Image(_Developing):
	"""
	Image Object capable of handling a single image
	"""
	def __init__ (self, path = False):
		"""
		Loads the Image on Initialization using cv2
		"""
		self.params = dict()
		self.name = path
		self.original = cv2.imread(os.path.join("..","..","data","images_training_rev1", path), 0)
		self.cur = self.original.copy()
		self.saved = {"orig":self.original}
	
	def toFile(self):
		cv2.imwrite(os.path.join("..","images","render", self.name), self.cur)

	def show(self, name = False):
		"""
		Visually loads the image
		"""
		if not name:
			cv2.imshow("Current Image", self.cur)
		else:
			cv2.imshow(name, self.saved[name])
		cv2.waitKey()

	def clearSaved(self):
		"""
		Clears the saved images
		"""
		self.saved = {"orig":self.original}

	def forceRange(self, low, high):
		imageMin, imageMax = self.cur.min(), self.cur.max()
		return (self.cur - imageMin)*((high - low)/(imageMax - imageMin)) + low

	def extreme(self, amp = .5):
		#0 < amp < 1
		print self.cur
		self.cur += (self.cur - 127) * amp
		self.cur[self.cur < 0] = 0
		print self.cur
		raw_input()

	def toColor(self):
		"""
		Paint dat image.
		"""
		return cv2.cvtColor(self.cur, cv.CV_GRAY2RGB)
	"""
	Pretty Printing for Image Object
	"""
	def __str__ (self):
		"""
		Resulting string when printed
		"""
		ret = (self.cur.shape[0], self.cur.shape[1], np.min(self.cur), np.max(self.cur))
		return "Image Numpy Array of size: (%d,%d)\nMax Value of %d\nMin Value of %d\n" % ret

	"""	
	Image like dictionary for attribute (params)
	"""
	def __getitem__(self, k):
		return self.params[k]

	def __setitem__(self, k, v):
		self.params[k] = v 

"""
Timer Wrapper
"""
def print_timing(func):
	def wrapper(*arg):
		t1 = time.time()
		res = func(*arg)
		t2 = time.time()
		print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
		return res
	return wrapper

"""
Development Wrapper
"""
def develop(func):
	def wrapper(*arg):
		image = Image("423000.jpg")
		image.load()
		func(image, *arg)
		image.save()
		return True
	return wrapper

def setParams(image):
	return image

"""
Image Processing
"""

"""
Removes Background
"""
def removeBackground(image, radius = 5):
	#Creating TopHat Kernel based on feature radius
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	scaled = image.forceRange(0,255)
	image.cur -= cv2.morphologyEx(scaled, cv2.MORPH_TOPHAT, kernel)
	return image

"""
Not Very Useful
"""
def mergeStatisticalRegions(image, seg = 1, thresh = 10):
	import _regions
	# blurred = cv2.GaussianBlur(image.cur, (0,0), 1.0)
	numRegions, average, regions = _regions.mergeStatisticalRegions(image.cur, seg)
	cv2.imshow("REGIONS", regions)
	cv2.waitKey()


"""
Edges by watershed
"""
def watershed(image, thresh = 18):
	erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
	dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
	
	_, binary = cv2.threshold(image.cur, thresh, 255, cv2.THRESH_BINARY)

	unsure = cv2.dilate(binary, dilateKernel)
	_, background = cv2.threshold(unsure, 1, 1, cv2.THRESH_BINARY_INV)

	eroded = cv2.erode(binary, erodeKernel)
	contours, _ = cv2.findContours(eroded, 
								   cv2.RETR_LIST, 
								   cv2.CHAIN_APPROX_SIMPLE)
	foreground = np.zeros_like(background, dtype=np.int32)
	cv2.drawContours(foreground, contours, -1, 2)

	markers = foreground + background
	cv2.watershed(image.toColor(), markers)

	select = image.cur.copy()
	select[markers != 2] = 0
	return select


@develop
@print_timing
def main(image):	
	image = setParams(image)
	# mergeStatisticalRegions(image)
	# image = removeBackground(image, radius = 10)
	image = removeBackground(image, radius = 10)
	image.cur = watershed(image, thresh = 20) #- cv2.GaussianBlur(image.cur, (0,0), 2)
	
def batchScript():
	for filename in os.listdir(os.path.join("..","..","data","images_training_rev1")):
		print "Rendering image", filename
		image = Image(filename)
		image = removeBackground(image, radius = 10)
		image.cur = watershed(image, thresh = 20)
		image.toFile()

if __name__ == "__main__":
	#main()
	batchScript()

