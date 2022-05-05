import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm.auto import tqdm
import scipy.sparse
from intramodes import *

#macroblock dimensions
blockw = 4
blockh = 4
blocksize = 4

DCTTransform = np.array([[1, 1, 1, 1],
						 [2, 1, -1, -2],
						 [1, -1, -1, 1],
						 [1, -2, 2, -1]])

IDCTTransform = np.array([[1, 1, 1, 0.5],
						 [1, 0.5, -1, -1],
						 [1, -0.5, -1, 1],
						 [1, -1, 1, -0.5]])

def luma4x4(Y):
	Yres = np.zeros(Y.shape)
	Ypred = np.zeros(Y.shape)

	#array to keep track of finished 4x4 macroblocks
	macroRows, macroCols = (Y.shape[0]//4, Y.shape[1]//4)
	available = [[False]*macroCols]*macroRows #mark as available once done
	modes = np.zeros((macroRows, macroCols))

	#process in 4x4 macroblocks
	for i in tqdm(range(0, Y.shape[0], 4), leave=False): #rows
		for j in range(0, Y.shape[1], 4): #cols
			iMac = i//4 #pun intended :^) macroblock index row
			jMac = j//4 #macroblock col
			samples = [False]*4 #ul, u, ur, l
			if iMac==0 and jMac==0: #first block
				samples = [False]*4 #ul, u, ur, l
			elif iMac==0: #top
				samples[3] = available[iMac][jMac-1]
			elif jMac==0: #left
				samples[1] = available[iMac-1][jMac]
				samples[2] = available[iMac-1][jMac+1]
			elif (jMac+1) == macroCols:
				samples[0] = available[iMac-1][jMac-1]
				samples[1] = available[iMac-1][jMac]
				samples[3] = available[iMac][jMac-1]
			else:
				samples[0] = available[iMac-1][jMac-1]
				samples[1] = available[iMac-1][jMac]
				samples[2] = available[iMac-1][jMac+1]
				samples[3] = available[iMac][jMac-1]

			#get neighbors
			if samples[0]:
				ul = Y[i-1, j-1]
			else:
				ul = 128

			if samples[1]:
				u = Y[i-1, j:j+4]
			else:
				u = np.ones(4) * 128

			if samples[2]:
				ur = Y[i-1, j+4:j+8]
			elif samples[1]:
				ur = np.ones(4) * Y[i-1, j+3]
			else:
				ur = np.ones(4) * 128

			if samples[3]:
				l = Y[i:i+4, j-1]
			else:
				l = np.ones(4) * 128

			bestpred = np.zeros((4,4))
			bestdiff = 16*255
			bestmode = 0
			
			temppred = vertical4x4(u)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 0
			
			temppred = horizontal4x4(l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 1
			
			temppred = dc4x4(u, l) #avg of up and left
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 2
			
			temppred = downleft4x4(u, ur)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 3
			
			temppred = downright4x4(ul, u, l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 4
			
			temppred = verticalright4x4(ul, u, l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 5
			
			temppred = horizontaldown4x4(ul, u, l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 6
			
			temppred = verticalleft4x4(u, ur)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 7
			
			temppred = horizontalup4x4(l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+4, j:j+4]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 8

			Yres[i:i+4, j:j+4] = Y[i:i+4, j:j+4] - bestpred
			Ypred[i:i+4, j:j+4] = bestpred
			modes[iMac,jMac] = bestmode
			available[iMac][jMac] = True

	return Yres, Ypred, modes

def luma16x16(Y):
	Yres = np.zeros(Y.shape)
	Ypred = np.zeros(Y.shape)

	#array to keep track of finished 4x4 macroblocks
	macroRows, macroCols = (Y.shape[0]//16, Y.shape[1]//16)
	available = [[False]*macroCols]*macroRows #mark as available once done
	modes = np.zeros((macroRows, macroCols))

	#process in 4x4 macroblocks
	for i in tqdm(range(0, Y.shape[0], 16), leave=False): #rows
		for j in range(0, Y.shape[1], 16): #cols
			iMac = i//16 #pun intended :^) macroblock index row
			jMac = j//16 #macroblock col
			samples = [False]*3 #ul, u, l
			if iMac==0 and jMac==0: #first block
				samples = [False]*3 #ul, u, l
			elif iMac==0: #top
				samples[2] = available[iMac][jMac-1]
			elif jMac==0: #left
				samples[1] = available[iMac-1][jMac]
			else:
				samples[0] = available[iMac-1][jMac-1]
				samples[1] = available[iMac-1][jMac]
				samples[2] = available[iMac][jMac-1]

			#get neighbors
			if samples[0]:
				ul = Y[i-1, j-1]
			else:
				ul = 128

			if samples[1]:
				u = Y[i-1, j:j+16]
			else:
				u = np.ones(16) * 128

			if samples[2]:
				l = Y[i:i+16, j-1]
			else:
				l = np.ones(16) * 128

			bestpred = np.zeros((16,16))
			bestdiff = 16*16*255
			bestmode = 0
			
			temppred = vertical16x16(u)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+16, j:j+16]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 0
			
			temppred = horizontal16x16(l)
			tempdiff = np.sum(np.abs(temppred-Y[i:i+16, j:j+16]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 1
			
			temppred = dc16x16(u, l) #avg of up and left
			tempdiff = np.sum(np.abs(temppred-Y[i:i+16, j:j+16]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpred = temppred
				bestmode = 2

			Yres[i:i+16, j:j+16] = Y[i:i+16, j:j+16] - bestpred
			Ypred[i:i+16, j:j+16] = bestpred
			modes[iMac,jMac] = bestmode
			available[iMac][jMac] = True

	return Yres, Ypred, modes

#do Cr and Cb together, choose same modes
def chroma8x8(Cr,Cb):
	Crres = np.zeros(Cr.shape)
	Cbres = np.zeros(Cb.shape)
	Crpred = np.zeros(Cr.shape)
	Cbpred = np.zeros(Cb.shape)

	#array to keep track of finished macroblocks
	macroRows, macroCols = (Cr.shape[0]//8, Cr.shape[1]//8)
	available = [[False]*macroCols]*macroRows #mark as available once done
	modes = np.zeros((macroRows, macroCols))

	#process in 4x4 macroblocks
	for i in tqdm(range(0, Cr.shape[0], 8), leave=False): #rows
		for j in range(0, Cr.shape[1], 8): #cols
			iMac = i//8 #pun intended :^) macroblock index row
			jMac = j//8 #macroblock col
			samples = [False]*3 #ul, u, l
			if iMac==0 and jMac==0: #first block
				samples = [False]*3 #ul, u, l
			elif iMac==0: #top
				samples[2] = available[iMac][jMac-1]
			elif jMac==0: #left
				samples[1] = available[iMac-1][jMac]
			else:
				samples[0] = available[iMac-1][jMac-1]
				samples[1] = available[iMac-1][jMac]
				samples[2] = available[iMac][jMac-1]

			#get neighbors
			if samples[0]:
				ulr = Cr[i-1, j-1]
				ulb = Cb[i-1, j-1]
			else:
				ulr = 128
				ulb = 128

			if samples[1]:
				ur = Cr[i-1, j:j+8]
				ub = Cbres[i-1, j:j+8]
			else:
				ur = np.ones(8) * 128
				ub = np.ones(8) * 128

			if samples[2]:
				lr = Cr[i:i+8, j-1]
				lb = Cb[i:i+8, j-1]
			else:
				lr = np.ones(8) * 128
				lb = np.ones(8) * 128

			bestpredr = np.zeros((8,8))
			bestpredb = np.zeros((8,8))
			bestdiff = 2*8*8*255
			bestmode = 0
			
			temppredr = vertical8x8(ur)
			temppredb = vertical8x8(ub)
			tempdiff = np.sum(np.abs(temppredr-Cr[i:i+8, j:j+8])) + np.sum(np.abs(temppredb-Cb[i:i+8, j:j+8]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpredr = temppredr
				bestpredb = temppredb
				bestmode = 0
			
			temppredr = horizontal8x8(lr)
			temppredb = horizontal8x8(lb)
			tempdiff = np.sum(np.abs(temppredr-Cr[i:i+8, j:j+8])) + np.sum(np.abs(temppredb-Cb[i:i+8, j:j+8]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpredr = temppredr
				bestpredb = temppredb
				bestmode = 1
			
			temppredr = dc8x8(ur, lr)
			temppredb = dc8x8(ub, lb)
			tempdiff = np.sum(np.abs(temppredr-Cr[i:i+8, j:j+8])) + np.sum(np.abs(temppredb-Cb[i:i+8, j:j+8]))
			if tempdiff < bestdiff:
				bestdiff = tempdiff
				bestpredr = temppredr
				bestpredb = temppredb
				bestmode = 2

			Crres[i:i+8, j:j+8] = Cr[i:i+8, j:j+8] - bestpredr
			Crpred[i:i+8, j:j+8] = bestpredr
			Cbres[i:i+8, j:j+8] = Cb[i:i+8, j:j+8] - bestpredb
			Cbpred[i:i+8, j:j+8] = bestpredb
			modes[iMac,jMac] = bestmode
			available[iMac][jMac] = True

	return Crres, Crpred, Cbres, Cbpred, modes

def intraframe(imgpath):
	#read in image and chroma subsample
	bgrimg = cv2.imread(imgpath)
	imshape = bgrimg.shape
	print(imshape[1])
	bgrimg = cv2.resize(bgrimg, (16*(imshape[1]//16), 16*(imshape[0]//16))) #resize to closest multiple of block for convenience
	imshape = bgrimg.shape
	YCrCbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2YCR_CB)
	Y,Cr,Cb =  cv2.split(YCrCbimg)
	fig = plt.figure()
	ax = fig.add_subplot(1, 3, 1)
	
	#ax.set_title("Original Luma Image")
	plt.imshow(Y, cmap="gray")
	ax.set_title("Original Cr Image")
	#plt.imshow(Cr, cmap="Reds")
	#plt.imshow(Cb, cmap="Blues")
	plt.axis("off")
	
	#ssv=2
	#ssh=2
	#Cr = cv2.boxFilter(Cr,ddepth=-1,ksize=(2,2))
	#Cb = cv2.boxFilter(Cb,ddepth=-1,ksize=(2,2))
	#Cr = Cr[::ssv,::ssh]
	#Cb  = Cb[::ssv,::ssh]
	print(Y.shape)
	print(Cr.shape)
	print(Cb.shape)
	
	
	Yres, Ypred, modes = luma4x4(Y)#luma4x4(Y)
	Crres, Crpred, Cbres, Cbpred, modeschrom = chroma8x8(Cr, Cb)

	#DCT
	# for i in tqdm(range(0, Y.shape[0], 4), leave=False): #rows
	# 	for j in range(0, Y.shape[1], 4): #cols
	# 		Yrestemp = Yres[i:i+4, j:j+4] 
	# 		Yrestemp = np.matmul(DCTTransform, Yrestemp)
	# 		Yrestemp = np.matmul(Yrestemp, DCTTransform.transpose())
	# 		Yres[i:i+4, j:j+4] = np.round(Yrestemp)

	print("sparsity:")
	print(1.0 - (np.count_nonzero(Yres)/float(Yres.size)))
	print("sparsity:")
	print(1.0 - (np.count_nonzero(Cbres)/float(Cbres.size)))	
	print("sparsity:")
	print(1.0 - (np.count_nonzero(Crres)/float(Crres.size)))	
	ax1 = fig.add_subplot(1, 3, 2)
	ax1.set_title("Image from Prediction")
	plt.imshow(Ypred, cmap="gray")
	#plt.imshow(Crpred, cmap="Reds")
	#plt.imshow(Cbpred, cmap="Blues")
	plt.axis("off")
	ax2 = fig.add_subplot(1, 3, 3)
	ax2.set_title("Stored Residuals")
	plt.imshow(Yres, cmap="gray")
	#plt.imshow(Crres, cmap="Reds")
	#plt.imshow(Cbres, cmap="Blues")
	plt.axis("off")
	plt.show()
	#np.save("compressed.npy", Yres.astype(np.int16))
	#np.save("intramodes.npy", modes.astype(np.int16))
	#scipy.sparse.save_npz('compressed.npz', scipy.sparse.csc_matrix(Yres.astype(np.int16)))
	#scipy.sparse.save_npz('intramodes.npz', scipy.sparse.csc_matrix(modes.astype(np.int16)))

	plt.imshow( cv2.cvtColor(
		cv2.cvtColor(
			np.dstack([Ypred, Crpred, Cbpred]).astype(np.uint8), cv2.COLOR_YCR_CB2BGR).astype(np.uint8),  
		cv2.COLOR_BGR2RGB))
	plt.axis("off")
	plt.show()

intraframe('../images/happy-corgi.jpg')