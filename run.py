"""
Image Processing for GalaxyZoo
author: sihrc
2/7/2014
"""
import os, cv2
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
		self.name = path
		if path:
			self.original = cv2.imread(os.path.join("..","images", path), 0)
			self.cur = self.original.copy()

	def show(self):
		"""
		Visually loads the image
		"""
		cv2.imshow("Rendered Image",self.cur)
		cv2.waitKey()

	def __str__ (self):
		"""
		Resulting string when printed
		"""
		ret = (self.cur.shape[0], self.cur.shape[1], np.min(self.cur), np.max(self.cur))
		return "Image Numpy Array of size: (%d,%d)\nMax Value of %d\nMin Value of %d\n" % ret

"""
Timer Wrapper
"""
def print_timing(func):
	def wrapper(*arg):
		t1 = time.time()
		res = func(*arg)
		t2 = time.time()
		print '%s took %0.3f ms' % (func.func.func_name, (t2-t1)*1000.0)
		return res
	return wrapper

"""
Development Wrapper
"""
def develop(name, func):
	@print_timing
	def wrapper(*arg):
		image = Image(name)
		image.load()
		ret = func(image, *arg)
		image.save()
		return ret
	return wrapper


@develop(name = "100008.jpg")
def main(image):
	image.log("Printing Image", image)	

if __name__ == "__main__":
	main()

