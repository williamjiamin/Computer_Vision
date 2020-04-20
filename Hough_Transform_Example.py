# 乐学偶得 版权所有 主讲人 ：William 公众号：乐学Fintech 官网：lexueoude.com
import numpy as np

from skimage.transform import hough_line, hough_circle_peaks
from skimage.feature import canny
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

# hard coding our image using numpy
image = np.zeros((200, 200))
# print(image)
index = np.arange(25, 175)
image[index[::-1], index] = 255
image[index, index] = 255

# 不断调整theta（jupyter noebook里），Precision 0.5度

every_tried_angles = np.linspace(-np.pi/2,np.pi/2,360)

hough_line()

# 画图1部分
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title('Our Hard Coding Image')

plt.tight_layout()
plt.show()
