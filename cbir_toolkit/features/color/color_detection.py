import numpy as np
import cv2
from matplotlib import pyplot as plt

def plot_rgb_colour_channels(image, boundaries=None):
	"""
	Function to detect and plot different colour channels in an image
	:param image: OpenCV BGR image
		Source image to be operated upon
	:param boundaries:
		Boundaries for each colour channel.
		Default as  [
						([17, 15, 100], [50, 56, 200]),
						([86, 31, 4], [220, 88, 50]),
						([25, 146, 190], [62, 174, 250]),
						([103, 86, 65], [145, 133, 128])
					]
	:return: no return
	"""

	if boundaries is None:
		boundaries = [
			([17, 15, 100], [50, 56, 200]),
			([86, 31, 4], [220, 88, 50]),
			([25, 146, 190], [62, 174, 250]),
			([103, 86, 65], [145, 133, 128])
		]

	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask=mask)

		# show the images
		cv2.imshow("images", np.hstack([image, output]))
		cv2.waitKey(0)
