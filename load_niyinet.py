from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.simpledatasetloader import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np 
import argparse
import cv2 


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to pre-trained model")
args = vars(ap.parse_args())


# initialize the class labels
classLabels = ["cat", "dog", "panda"]


