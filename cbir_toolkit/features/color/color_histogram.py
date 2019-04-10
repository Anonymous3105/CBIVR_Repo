import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_color_histogram(image, channels, mask=None, hist_size=(256), ranges=(0, 256)):
	"""
	Function to get the histogram of provided image in the channel required

	:param image:  source image of type uint8 or float32
	:param channels:  index of channel for which histogram to be calculated. Given in square brackets[],
		If input is grayscale image, its value is [0].
		For color image, you can pass [0],[1] or [2] to calculate histogram of blue,green or red channel respectively.
	:param mask: mask image. default as None to compute histogram of full image.
	:param hist_size: Bin count of the histogram. Need to be given in square brackets. Default as [256]
	:param ranges: the range of intensity values you want to measure. Default as [0, 256]
	:return: hist is a 256x1 array, each value corresponds to number of pixels in that image with its corresponding pixel value.
	"""

	return cv2.calcHist([image], channels, mask, hist_size, ranges)


def plot_gray_histogram(image, mask=None, hist_size=256, ranges=(0, 256)):
	"""
	Function to plot the histogram of the gray channel of the provided grayscale image

	:param image: source grayscale image of type uint8 or float32
	:param mask:  mask image. default as None to compute histogram of full image.
	:param hist_size: Bin count of the histogram. Need to be given in square brackets. Default as [256]
	:param ranges: the range of intensity values you want to measure. Default as [0, 256]
	:return: no return
	"""

	if mask is None:
		plt.hist(image.ravel(), hist_size, ranges)
	else:
		hist_mask = cv2.calcHist(image, [0], mask, hist_size, ranges)
		plt.plot(hist_mask)

	plt.title("Grayscale Histogram of image")
	plt.xlabel("Intensity values")
	plt.ylabel("Count of pixels")

	plt.xlim(ranges)
	plt.show()


def plot_rgb_histogram(image, mask, hist_size=256, ranges=(0, 256)):
	"""
	Function to plot the histogram RGB channel values of the provided image

	:param image: source image of type uint8 or float32
	:param mask: mask image. default as None to compute histogram of full image.
	:param hist_size: Bin count of the histogram. Need to be given in square brackets. Default as [256]
	:param ranges: the range of intensity values you want to measure. Default as [0, 256]
	:return: no return
	"""

	color = ('b', 'g', 'r')
	for i, col in enumerate(color):
		hist = cv2.calcHist([image], [i], mask, hist_size, ranges)
		plt.plot(hist, color=col)
		plt.xlim([0, 256])

	plt.title("RGB Histogram of image")
	plt.xlabel("Intensity values")
	plt.ylabel("Count of pixels")
	plt.legend(loc=1, labels=["Blue", "Red", "Green"])

	plt.xlim(ranges)
	plt.show()


def plot_2d_rgb_color_histograms(image, mask=None, hist_size=(8, 8, 8), ranges=(0, 256, 0, 256, 0 ,256)):
	"""
	:param image: OpenCV BGR Image
		Source image to be operated upon
	:param mask: mask image. default as None to compute histogram of full image.
	:param hist_size: Bin count of the histogram. Need to be given in square brackets. Default as [8, 8, 8]
	:param ranges: the range of intensity values you want to measure. Default as [0, 256, 0, 256, 0, 256]
	:return: no return values
	"""

	chans = cv2.split(image)
	fig = plt.figure()

	# plot a 2D color histogram for green and blue
	ax = fig.add_subplot(131)
	hist = cv2.calcHist([chans[1], chans[0]], [0, 1], mask, hist_size[0:2], ranges[0:4])
	p = ax.imshow(hist, interpolation="nearest")
	ax.set_title("2D Color Histogram for Green and Blue")
	plt.colorbar(p)

	# plot a 2D color histogram for green and red
	ax = fig.add_subplot(132)
	hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, hist_size[1:], ranges[2:])
	p = ax.imshow(hist, interpolation="nearest")
	ax.set_title("2D Color Histogram for Green and Red")
	plt.colorbar(p)

	# plot a 2D color histogram for blue and red
	ax = fig.add_subplot(133)
	hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, hist_size[0:1]+hist_size[1:2], ranges[0:2]+ranges[4:])
	p = ax.imshow(hist, interpolation="nearest")
	ax.set_title("2D Color Histogram for Blue and Red")
	plt.colorbar(p)

	print("2D histogram wavelet: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))
	plt.show()


def get_rgb_histogram_features(image, channels=(0, 1, 2), mask=None, hist_size=(8, 8, 8), ranges=(0, 256, 0, 256, 0, 256)):
	"""
	Function to get histogram features of a OpenCV BGR Image.

	:param image: source image of type uint8 or float32
	:param channels: index of channel for which histogram to be calculated. Given in square brackets[], Default: [0, 1, 2]
		If input is grayscale image, its value is [0].
		For color image, you can pass [0],[1] or [2] to calculate histogram of blue,green or red channel respectively.
	:param mask: mask image. default as None to compute histogram of full image.
	:param hist_size: Bin count of the histogram. Need to be given in square brackets. Default as [8, 8, 8]
	:param ranges: the range of intensity values you want to measure. Default as [0, 256, 0, 256, 0, 256]
	:return: returns a normalized flattened histogram numpy array
	"""

	hist = cv2.calcHist([image], channels, mask, hist_size, ranges)
	hist = cv2.normalize(hist, dst=np.array([])).flatten()

	return hist
