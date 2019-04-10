import cv2
import numpy as np
from pywt import wavedec2
from scipy.special import entr


def get_lastlevel_DWT(image, wavelet, lev=3):
	return wavedec2(image, wavelet, level=lev)[:2]


def extract_features(arr):
	features = []
	for a in arr:
		features.extend([np.mean(a), np.std(a), entr(a).sum()])

	for i in range(len(features)):
		if features[i] == np.NAN:
			features[i] = 0

	return features


def get_BGR_DWT_features(image, wavelet='db1'):
	"""
	Function to compute 3-level DWT RGB planes of an image. The function operates on the following algorithm:
		1. Split the RGB image into individual planes
		2. For each color plane, decompose the plane using Discrete Wavelet Transform (DWT) to the 3rd level
		3. For each obtained band in the 3rd level decomposition compute features like mean, standard deviation & entropy.
		4. Follow the same procedure for the Green and Blue color.

	:param image: OpenCV image on ndarray_like
	:param wavelet: (Default = 'db1'; Others include 'haar', 'db2'
		Algorithm to be used to compute the DWT of image
	:return:
	"""
	image_resized = cv2.resize(image, (256, 256))
	split_images = cv2.split(image_resized)
	bgr_features = []

	for image_ch in split_images:
		An, (cHn, cVn, cDn) = get_lastlevel_DWT(image_ch, wavelet)
		bgr_features.extend(extract_features([An, cHn, cVn, cDn]))

	return np.array(bgr_features)
