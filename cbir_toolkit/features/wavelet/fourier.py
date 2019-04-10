import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_fourier_features(image):
	"""
	Function to plot edges detected by using Fast Fourier Transform
	:param image:
	:return:
	"""
	rows, cols = image.shape[:2]
	crow,ccol = rows // 2, cols // 2

	f = np.fft.fft2(image)
	fshift = np.fft.fftshift(f)
	fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)

	plt.subplot(131)
	plt.imshow(image, cmap='gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(132)
	plt.imshow(img_back, cmap='gray')
	plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
	plt.subplot(133)
	plt.imshow(img_back)
	plt.title('Result in JET'), plt.xticks([]), plt.yticks([])


def get_fourier_descriptors(image):
	""" Function to find and return the	Fourier-Descriptor of the image contour
	:param image: OpenCV uint8 or float32 array_like
		Source image to compute the fourier descriptors on
	:return: array_like
	"""

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(thresh, image_binary) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


	contour = []
	_, contour, hierarchy = cv2.findContours(
		image_binary,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_NONE,
		contour)

	contour_array = contour[0][:, 0, :]
	contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
	contour_complex.real = contour_array[:, 0]
	contour_complex.imag = contour_array[:, 1]
	fourier_result = np.fft.fft(contour_complex)

	return fourier_result


def truncate_descriptor(descriptors, degree):
	"""
	Function to truncate an unshifted fourier descriptor array and returns one also unshifted
	:param descriptors: array_like
		Feature descriptors extracted from an image
	:param degree: int
		Degree to which features need to be shifted
	:return: array_like
		Truncated feature array
	"""
	descriptors = np.fft.fftshift(descriptors)
	center_index = len(descriptors) // 2
	descriptors = descriptors[center_index - degree // 2:center_index + degree // 2]
	descriptors = np.fft.ifftshift(descriptors)

	return descriptors


def reconstruct_image(descriptors, degree, plot_contours=False):
	""" Funtion that attempts to reconstruct the image using the first [degree] descriptors of descriptors
	:param descriptors: array_like
		Fourier descriptors extracted from an image
	:param degree: int
		The degree of extracted fourier descriptors
	:param plot_contours: bool, optional (Default= False)
		Boolean to decide whether to plot the reconstructed image or not
	:return OpenCV array_like of uint8
	"""
	# truncate the long list of descriptors to certain length
	descriptor_in_use = truncate_descriptor(descriptors, degree)
	contour_reconstruct = np.fft.ifft(descriptor_in_use)
	contour_reconstruct = np.array([contour_reconstruct.real, contour_reconstruct.imag])
	contour_reconstruct = np.transpose(contour_reconstruct)
	contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)

	# make positive
	if contour_reconstruct.min() < 0:
		contour_reconstruct -= contour_reconstruct.min()

	# normalization
	contour_reconstruct *= 800 // contour_reconstruct.max()

	# type cast to int32
	contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
	black = np.zeros((800, 800), np.uint8)

	# draw and visualize
	if plot_contours:
		cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=-1)
		cv2.imshow("black", black)
		cv2.waitKey(1000)
		cv2.imwrite("reconstruct_result.jpg", black)
		cv2.destroyAllWindows()

	return descriptor_in_use
