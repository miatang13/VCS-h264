import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

bgrimg = cv2.imread('../images/catstairs.png')
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title("Original Image")
plt.imshow(cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB))
plt.axis("off")
#plt.show()
imshape = bgrimg.shape
bgrimg = cv2.resize(bgrimg, (8*imshape[1]//8, 8*imshape[0]//8)) #resize to closest multiple of 8 for convenience
imshape = bgrimg.shape
#plt.imshow(bgrimg)
#plt.show()
YCrCbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2YCR_CB)
Y,Cr,Cb =  cv2.split(YCrCbimg)
Y = np.array(Y).astype(np.int16) - 128
Cr = np.array(Cr).astype(np.int16) - 128
Cb = np.array(Cb).astype(np.int16) - 128
channels = [Y,Cr,Cb]

blocksize = 8

def cuHelper(ind):
	if ind == 0:
		return 1/math.sqrt(2)
	else:
		return 1

#reference: https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
def dct(inputBlock):
	result = np.zeros((blocksize, blocksize))
	for i in range(blocksize):
		for j in range(blocksize):
			cosSum = 0
			for k in range(blocksize):
				for l in range(blocksize):
					temp = inputBlock[k][l] * math.cos((2*k+1)*i*math.pi/(2*blocksize)) * math.cos((2*l+1)*j*math.pi/(2*blocksize))
					cosSum += temp
			result[i][j] = (1/math.sqrt(2*blocksize)) * cuHelper(i) * cuHelper(j) * cosSum
	return result

def dctTest():
	testMatrix = np.ones((blocksize,blocksize)) * 255
	print("original matrix:")
	print(testMatrix)
	print("DCT Transform of matrix")
	result = dct(testMatrix)
	print(result)

dctTest()

def dctMatrix():
	result = np.zeros((blocksize, blocksize))
	for i in range(blocksize):
		for j in range(blocksize):
			if i==0:
				result[i,j] = 1/math.sqrt(blocksize)
			else:
				result[i,j] = math.sqrt(2/blocksize) * math.cos((2*j+1)*i*math.pi/(2*blocksize))
	return result

dctmat = dctMatrix()
print(dctmat)

#apply dct transform with matrix multiplication and return transformed image
def dct2(matrix):
	dctmat = dctMatrix()
	result = np.matmul(dctmat, matrix)
	result = np.matmul(result, dctmat.transpose())
	return result

def idct2(matrix):
	dctmat = dctMatrix()
	result = np.matmul(dctmat.transpose(), matrix)
	result = np.matmul(result, dctmat)
	return result

def dct2Test():
	#testMatrix = np.ones((blocksize,blocksize)) * 255
	testMatrix=np.array([[26,-5,-5,-5,-5,-5,-5,8],
             			[64,52,8,26,26,26,8,-18],
            			[126,70,26,26,52,26,-5,-5],
             			[111,52,8,52,52,38,-5,-5],
             			[52,26,8,39,38,21,8,8],
             			[0,8,-5,8,26,52,70,26],
             			[-5,-23,-18,21,8,8,52,38],
             			[-18,8,-5,-5,-5,8,26,8]]) #verifying example from reference
	print("original matrix:")
	print(testMatrix)
	print("DCT Transform of matrix using matrix multiplication")
	result = dct2(testMatrix)
	print(result)

dct2Test()

#jpeg standard quantization matrices for lum and chrom
QY=np.array([[16,11,10,16,24,40,51,61],
             [12,12,14,19,26,48,60,55],
             [14,13,16,24,40,57,69,56],
             [14,17,22,29,51,87,80,62],
             [18,22,37,56,68,109,103,77],
             [24,35,55,64,81,104,113,92],
             [49,64,78,87,103,121,120,101],
             [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
             [18,21,26,66,99,99,99,99],
             [24,26,56,99,99,99,99,99],
             [47,66,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99],
             [99,99,99,99,99,99,99,99]])

QF=50.0 #quality factor, must be between 1 and 99
if QF<50 and QF>1:
	scale = 50/QF
elif QF<100:
	scale = (100-QF)/50
else:
	print("Invalid quality setting, must be between 1 and 99.")
Q=[np.clip(np.round(QY*scale), 1, 255),
   np.clip(np.round(QC*scale), 1, 255),
   np.clip(np.round(QC*scale), 1, 255)]


def completeDCT(): #used to pass in image to process 1 channel
	compressed = []
	for channel in range(3):
		image = channels[channel]
		result = np.zeros(image.shape)
		for i in range(0, imshape[0], blocksize):
			for j in range(0,imshape[1], blocksize):
				block = image[i:i+blocksize, j:j+blocksize]
				d = dct2(block) #perform transform
				d = np.round(np.divide(d, Q[channel])) #perform quantization
				result[i:i+blocksize, j:j+blocksize] = d
		#ax = fig.add_subplot(1,2,2)
		#ax.set_title("Image as 8x8 DCT blocks")
		#plt.imshow(result, cmap='gray', vmax = np.max(result)*0.01, vmin=0)
		#plt.axis("off")
		#plt.show()
		compressed.append(result)
	#decompression
	decompressed = []
	for channel in range(3):
		image = compressed[channel]
		result = np.zeros(image.shape)
		for i in range(0, imshape[0], blocksize):
			for j in range(0,imshape[1], blocksize):
				block = image[i:i+blocksize, j:j+blocksize]
				d = np.multiply(block, Q[channel]) #perform de-quantization
				d = idct2(d) #inverse dct
				result[i:i+blocksize, j:j+blocksize] = d
		decompressed.append(result)
	newYCrCb = np.dstack(decompressed)
	print(newYCrCb.shape)
	newBGR = cv2.cvtColor(np.float32(newYCrCb), cv2.COLOR_YCR_CB2BGR)
	ax = fig.add_subplot(1,2,2)
	ax.set_title("Compressed Image")
	plt.imshow(cv2.cvtColor(newBGR, cv2.COLOR_BGR2RGB))
	plt.axis("off")
	plt.show()

#completeDCT(Y)
completeDCT()
