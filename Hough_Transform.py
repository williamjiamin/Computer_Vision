# 乐学偶得 版权所有 主讲人 ：William 公众号：乐学Fintech 官网：lexueoude.com
from skimage import io, color
from skimage.transform import (hough_line, probabilistic_hough_line)
from skimage.feature import canny

image = io.imread("Trevor-Noah.jpg")
image = color.rgb2grey(image)

edges = canny(image, 2, 1, 25)

# 找到edges之后，进行霍夫变换
lines = hough_line(image)
probabilistic_line=probabilistic_hough_line(edges)
print(lines)

