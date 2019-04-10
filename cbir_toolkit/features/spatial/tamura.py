import numpy as np
import cv2
from skimage.feature.texture import greycomatrix
from scipy.stats import moment


def get_tamura_features(image):
	"""
	Function to get Tamura features of an image
	:param image: OpenCV grayscale image ndarray like
	:return: Returns a dictionary of features (coarseness, contrast, directionality)

	Note: Line-Likeness features has not been returned as the feature is still under development
			and can cause issues if left unhandled.
	"""
	features = dict()

	features["coarseness"] = get_coarseness_tamura(image)
	features["contrast"] = get_contrast(image)
	features["directionality"] = get_directionality(image)
	# features["linelikeness"] = get_linelikeness(image)

	return features


def get_coarseness_tamura(image):

	assert image.shape[0] > 64 and image.shape[1] >= 64, "Image dimensions should be minimum 64X64"

	image = cv2.resize(image, (1024,1024))
	H, W = image.shape[:2]
	Ei = []
	SBest = np.zeros((H, W))

	for k in range(1, 7):
		Ai = np.zeros((H, W))
		Ei_h = np.zeros((H, W))
		Ei_v = np.zeros((H, W))

		for h in range(2**(k-1)+1, H-(k-1)):
			for w in range(2**(k-1)+1, W-(k-1)):
				image_subset = image[h-(2**(k-1)-1): h+(2**(k-1)-1)-1, w-(2**(k-1)-1): w+(2**(k-1)-1)-1]
				Ai[h, w] = np.sum(image_subset)

		for h in range(2**(k-1)+1, H-k):
			for w in range(2 ** (k - 1) + 1, W-k):
				try:
					Ei_h[h, w] = Ai[h+(2**(k-1)-1), w] - Ai[h-(2**(k-1)-1), w]
					Ei_v[h, w] = Ai[h, w+(2**(k-1)-1)] - Ai[h, w-(2**(k-1)-1)]
				except IndexError:
					pass

		Ei_h /= 2 ** (2 * k)
		Ei_v /= 2 ** (2 * k)

		Ei.append(Ei_h)
		Ei.append(Ei_v)

	Ei = np.array(Ei)
	for h in range(H):
		for w in range(W):
			maxv_index = np.argmax(Ei[:, h, w])
			k_temp = (maxv_index + 1) // 2
			SBest[h, w] = 2**k_temp

	coarseness = np.sum(SBest) / (H * W)
	return coarseness


def get_contrast(image, mask=None, n=0.25):

	H, W = image.shape[:2]
	hist = cv2.calcHist(image, [0], mask, [256], [0,256])
	levels = 256

	count_probs = hist / (H * W)

	std = np.std(count_probs)
	moment_4th = moment(count_probs, 4)
	kurtosis = moment_4th / (std ** 4)

	contrast = std / (kurtosis ** n)
	return contrast


def get_directionality(image, threshold=12):

	# TODO: Find the actual threshold to eliminate value of 12
	H, W = image.shape[:2]

	kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
	kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

	img_prewittx = cv2.filter2D(image, -1, kernelx)
	img_prewitty = cv2.filter2D(image, -1, kernely)

	deltas = 0.5 * (np.abs(img_prewittx) + np.abs(img_prewitty))
	try:
		angles = np.arctan(img_prewitty / img_prewittx) + (np.pi / 2)
	except ZeroDivisionError:
		pass
	np.nan_to_num(angles)

	bin_angles = np.array(range(0, 180, 20)) * np.pi / 180
	dir_vector = np.zeros(9)

	digitized_angles = np.digitize(angles, bin_angles)

	for h in range(H):
		for w in range(W):
			if deltas[h, w] > threshold:
				dir_vector[digitized_angles-1] += 1

	return dir_vector


def get_linelikeness(image):
	#  TODO: Check this functions working. If it works properly or not
	#  Possible Runtime warnings

	H, W = image.shape[:2]

	kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
	kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

	img_prewittx = cv2.filter2D(image, -1, kernelx)
	img_prewitty = cv2.filter2D(image, -1, kernely)
	angles = np.arctan(img_prewitty / img_prewittx) + (np.pi / 2)

	n_bins = 9
	bin_angles = np.array(range(0, 180, 20)) * np.pi / 180

	digitized_angles = np.digitize(angles, bin_angles)

	comat = greycomatrix(digitized_angles, [1], [0, np.pi/2], levels=256)

	line_likeness = 0
	for i in range(n_bins):
		for j in range(n_bins):
			line_likeness += comat[i, j] * np.cos((i-j) * 2 * np.pi / n_bins)

	line_likeness /= np.sum(comat)

	return line_likeness
