import os
from glob import * 
import random
import json
import warnings

from IPython.display import Markdown, display

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import skimage
import PIL
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
from PIL import Image 
import cv2

import numpy as np
import pandas as pd

from helpers.nms import non_max_suppression
from helpers.iou import compute_iou
from helpers.plot_embedding_images import plot_embedding_images
from helpers.plot_history import plot_history

import tensorflow as tf
import keras

from keras.models import Model
from keras import backend as K

from tensorflow.keras.applications import DenseNet121
from tf_explain.core.activations import ExtractActivations
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import * 
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


#Reproductibility
seed_value= 1
random.seed(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = str(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#Filter Warnings
warnings.simplefilter("ignore")