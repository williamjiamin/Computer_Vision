from skimage.transform import probabilistic_hough_line
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import canny

image = data.camera()
edges = canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=10,
                                 line_length= 10, line_gap= 3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Original input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Edges of the image using canny')


ax[2].imshow(edges * 0)

for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0],p1[0]),(p0[1],p1[1]))

ax[2].set_title('Probabilistic Hough Transform')

plt.tight_layout()
plt.show()
plt.show()

# 1.Randomly, select a new point for voting in the accumulator array, with contributions to all available bins
# (as referenced in [32], bin stands for a pair of (λ, θ)). Then remove the selected pixel from the input image.
#
# 2.Check if the highest peak (the pair of (λ, θ) with the most voting points) in the updated accumulator is greater
# than a pre-defined threshold th(N). If not then go to Step 1.
#
# 3.Find all lines with the parameter (λ, θ) which was specified by the peak in Step 2. Choose the longest segment
# (which can be denoted by starting point Pt0 and ending point Pt1) of all lines.
#
# 4.Remove all the points of the longest line from the input image.
#
# 5.Remove all the points of the selected line in Step 3 (Pt0−Pt1) from the accumulator,
# which means those points do not attend any other voting process.
#
# 6.If the selected segment is longer than a pre-defined minimum length,
# then take the segment (Pt0−Pt1) as one of the output results.
#
# Go to Step 1.
