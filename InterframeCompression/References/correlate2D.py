from imageio import imread
from matplotlib import pyplot as plt
import scipy.signal as sp
import numpy as np

img = imread('https://i.stack.imgur.com/JL2LW.png', pilmode='L')
temp = imread('https://i.stack.imgur.com/UIUzJ.png', pilmode='L')

corr = sp.correlate2d(img - img.mean(), 
                      temp - temp.mean(),
                      boundary='symm',
                      mode='full')

# coordinates where there is a maximum correlation
max_coords = np.where(corr == np.max(corr))
y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match

# plotting
fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
                                                    figsize=(6, 15))
ax_orig.imshow(img, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_template.imshow(temp, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()
ax_corr.imshow(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_corr.plot(max_coords[1], max_coords[0], 'c*', markersize=5)
ax_orig.plot(x, y, 'ro')
plt.show()
                                                    
# plt.imshow(img, cmap='gray')
# plt
# plt.plot(max_coords[1], max_coords[0],'c*', markersize=5)
# plt.imshow(corr, cmap='gray')
# plt.show()

print("finished")