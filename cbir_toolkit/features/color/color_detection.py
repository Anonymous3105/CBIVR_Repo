import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_rgb_colour_channels(image, boundaries=None):
	"""
	Function to detect and plot different colour channels in an image
	:param image: OpenCV BGR image
		Source image to be operated upon
	:return: no return
	"""

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	w, h = image.shape[:2]

	plt.subplot("221")
	plt.imshow(image)
	plt.title("Input Image")
	plt.show()

	image_r = image.copy()
	image_r[:, :, 1] = 0
	image_r[:, :, 2] = 0
	plt.subplot("222")
	plt.title("Red channel of Image")
	plt.imshow(image_r)

	image_g = image.copy()
	image_g[:, :, 0] = 0
	image_g[:, :, 2] = 0
	plt.subplot("223")
	plt.title("Green channel of Image")
	plt.imshow(image_g)

	image_b = image.copy()
	image_b[:, :, 0] = 0
	image_b[:, :, 1] = 0
	plt.subplot("224")
	plt.title("Blue channel of Image")
	plt.imshow(image_b)

	plt.show()

