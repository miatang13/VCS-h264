# import the cv2 library
import cv2

# The function cv2.imread() is used to read an image.
img = cv2.imread('test.jpg')
imgYYC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

# We do 2x2 subsampling
sampRate = 2
ksize = (2, 2)

# Box filter over vertical and horizontal 
cr = cv2.boxFilter(imgYYC[:,:,1],ddepth=-1,ksize=ksize)
cb = cv2.boxFilter(imgYYC[:,:,2],ddepth=-1,ksize=ksize)

# Subsample
crSamples = cr[::sampRate, ::sampRate]
cbSamples = cr[::sampRate, ::sampRate]

# Final subsampled
subsampledImg = [imgYYC[:,:,0], crSamples, cbSamples]


