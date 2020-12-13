import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

# image = img_as_ubyte(data.coins()[160:230, 70:270])
image = data.coins()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))

ax.imshow(image)
plt.show()
