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


import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import data,feature

#使用Probabilistic Hough Transform.
image = data.camera()
edges = feature.canny(image, sigma=2, low_threshold=1, high_threshold=25)
lines = st.probabilistic_hough_line(edges, threshold=10, line_length=5,line_gap=3)

# 创建显示窗口.
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
plt.tight_layout()

#显示原图像
ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()

#显示canny边缘
ax1.imshow(edges, plt.cm.gray)
ax1.set_title('Canny edges')
ax1.set_axis_off()

#用plot绘制出所有的直线
ax2.imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
row2, col2 = image.shape
ax2.axis((0, col2, row2, 0))
ax2.set_title('Probabilistic Hough')
ax2.set_axis_off()
plt.show()
