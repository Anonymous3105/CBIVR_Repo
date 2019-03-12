try:
	import numpy as np
	import cv2
	from matplotlib import pyplot as plt
except ImportError as error:
	print(error.__class__.__name__ + ": " + error.message)
