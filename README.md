# CBIR Toolkit (cbtk)

## Utility
A Python library which encompasses all commonly used CBIR functions, some cutting edge feature extraction functions which are difficult to program and a built in ranker to ensure that is it easily available, easily utilized and easy to create complex systems with.


## Installation Instructions
Installation is rather straightforward. 
+ Create a python 3 virtual environment via the ```virtualenv env``` command to prevent any mismatch or multiple copies of the same dependency modules. Enter this virtual environment whenever using ```cbtk```. 
+ Upon extracting the source files from the ```.zip``` archive, running ```pip3 install .``` will install a local copy of the module in your virtual environment.
+ The module will be available to use by executing ```import cbir_toolkit``` in python.

## Currently Implemented Features
- Texture
    - Gray Level Co-occurrence Matrix
    - GLCM Features
    - LBP
    - Haralick Features
- Color
    - Pixel Values Histogram
- Keypoints
    - HOG Descriptors
    - SIFT Features
    - SURF Features
    - ORB Features
- Wavelet
    - BGR DWT Features
    - Fourier Descriptors
- Spatial
    - Color Correlogram
    - Color Autocorrelogram
    - Gabor Features
    - Tamura Features
        - Coarseness
        - Contrast
        - Directionality
        - Linelikeness
- Metrics
    - Cosine Similarity
    - Pearson Correlation Coefficient
    - Histogram Intersection
    - Jaccard Similarity
    - Absolute Distance
    - Sum of Squares of Absolute Differences
    - Euclidean Distance
    - City Block Distance
    - Canberra Distance
    - Maximum Value Distance
    - Minkowski Distance
    - Chi-Squared Distance
    - Hamming Distance
    - Wassertein Distance
- Tools
    - Loader
    - Ranker

## Dependencies
+ opencv-contrib-python==3.4.2.16
+ matplotlib
+ scikit-image
+ scipy
+ mahotas
+ Pywavelets
