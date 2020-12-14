import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

# image = img_as_ubyte(data.coins()[160:230, 70:270])
image = data.coins()
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

hough_radii = np.arange(10, 35, 2)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=20)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.grey2rgb(image)

for center_y, center_x, radius in zip(cy, cx, radii):
    circle_y, circle_x = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    image[circle_y, circle_x] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
# ax.imshow(edges)
plt.show()
