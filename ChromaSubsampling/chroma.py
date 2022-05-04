import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../images/happy-corgi.jpg')
rows,cols,channels = img.shape

# TODO: We do BGR to YCRCB manually
imgYYC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

# We do 4:2:0 subsampling
sampRate = 2
ksize = (2, 2)

# Box filter over vertical and horizontal 
cr = cv2.boxFilter(imgYYC[:,:,1],ddepth=-1,ksize=ksize)
cb = cv2.boxFilter(imgYYC[:,:,2],ddepth=-1,ksize=ksize)

# Subsample
crSamples = cr[::sampRate, ::sampRate]
cbSamples = cb[::sampRate, ::sampRate]

# Final subsampled
subsampledImg = [imgYYC[:,:,0], crSamples, cbSamples]
print(len(subsampledImg[0]), len(subsampledImg[1]))

# Convert YCrCb back to BGR for display
finalImg = np.zeros((rows, cols, channels), dtype = np.uint8)
for i in range(rows):
    for j in range(cols):
        Y = subsampledImg[0][i, j]
        Cr = crSamples[int(i/sampRate), int(j/sampRate)]
        Cb = cbSamples[int(i/sampRate), int(j/sampRate)]
        r = Y + 1.4022 * (Cr - 128)
        g = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
        b = Y + 1.77200 * (Cb - 128)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        finalImg[i, j] = [b, g, r]

# show image with matplotlib
plt.figure(figsize=(1, 2)) # specifying the overall grid size
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB)]
labels = ["Input Image", "Output Image"]
for i in range(2):
    plt.subplot(1,2,i+1)    # the number of images in the grid is 1*2 (2)
    plt.imshow(images[i])
    plt.xlabel(labels[i])

plt.show()

# Output image
cv2.imwrite("Output.jpg", finalImg)
print("Finished writing image")