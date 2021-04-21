import numpy as np
import cv2
import sys


img = cv2.imread(sys.argv[1])
print(img.shape)
