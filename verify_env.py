import sys
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import cv2
import matplotlib
import seaborn
import arff
import ucimlrepo

print("=" * 50)
print("ENVIRONMENT VERIFICATION")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {seaborn.__version__}")
print(f"Liac-ARFF: {arff.__version__ if hasattr(arff, '__version__') else 'Loaded'}")
print(f"UCIMLRepo: {ucimlrepo.__version__ if hasattr(ucimlrepo, '__version__') else 'Loaded'}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Available: {gpus}")
if len(gpus) == 0:
    print("TensorFlow GPU not detected, falling back to CPU execution.")
print("=" * 50)
print("ALL IMPORTS VERIFIED SUCCESSFULLY!")
print("=" * 50)
