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

plt.plot(max_coords[1], max_coords[0],'c*', markersize=5)
plt.imshow(corr, cmap='hot')
plt.show()

print("finished")