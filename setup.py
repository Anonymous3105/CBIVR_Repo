from setuptools import setup

setup(
    name='cbir_toolkit',
    version='0.16',
    description='Content Based Image Retrieval Toolkit',
    url='https://github.com/Anonymous3105/CBIVR_Repo',
    packages=['cbir_toolkit'],
    install_requires=[
        'opencv-contrib-python==3.4.2.16',
        'matplotlib',
        'scikit-image',
        'scipy',
        'mahotas',
        'Pywavelets'
    ]
)