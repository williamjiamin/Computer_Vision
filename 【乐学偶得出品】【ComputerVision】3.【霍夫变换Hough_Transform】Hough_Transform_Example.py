# 乐学偶得 版权所有 主讲人 ：William 公众号：乐学Fintech 官网：lexueoude.com
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
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

every_tried_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)

h, theta, d = hough_line(image, theta=every_tried_angles)

# 画图1部分
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title('Our Hard Coding Image')

ax[1].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]),
                                    np.rad2deg(theta[0]), d[-1], d[0]], aspect=1 / 1.5)

ax[1].set_title('Hough Transform')
ax[1].set_xlabel('Angles(in degree)')
ax[1].set_ylabel('Distance(in pixels)')

# 画出原图
ax[2].imshow(image, cmap=cm.gray)

origin = np.array((0, image.shape[1]))

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    # 极坐标转换为笛卡尔坐标
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')

ax[2].set_xlim(origin)
ax[2].set_ylim(image.shape[0], 0)
ax[2].set_title('Detected Lines')

plt.tight_layout()
plt.show()


