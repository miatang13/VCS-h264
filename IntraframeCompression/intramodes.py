import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm.auto import tqdm

def vertical4x4(u):
	pred = np.zeros((4,4))
	for i in range(4):
		pred[i] = u
	return pred

def horizontal4x4(l):
	pred = np.zeros((4,4))
	for i in range(4):
		pred[:,i] = l
	return pred

def dc4x4(u, l):
	pred = np.ones((4,4))
	avg = np.sum(u+l) // 8
	pred = pred * avg
	return pred

def downleft4x4(u, ur):
	pred = np.zeros((4,4))
	pred[0,0] = u[0]//4 + u[1]//2 + u[2]//4
	pred[0,1] = u[1]//4 + u[2]//2 + u[3]//4
	pred[1,0] = pred[0,1]
	pred[0,2] = u[2]//4 + u[3]//2 + ur[0]//4
	pred[1,1] = pred[0,2]
	pred[2,0] = pred[0,2]
	pred[0,3] = u[3]//4 + ur[0]//2 + ur[1]//4
	pred[1,2] = pred[0,3]
	pred[2,1] = pred[0,3]
	pred[3,0] = pred[0,3]
	pred[1,3] = ur[0]//4 + ur[1]//2 + ur[2]//4
	pred[2,2] = pred[1,3]
	pred[3,1] = pred[1,3]
	pred[2,3] = ur[1]//4 + ur[2]//2 + ur[3]//4
	pred[3,2] = pred[2,3]
	pred[3,3] = ur[2]//4 + 3*ur[3]//4
	return pred

def downright4x4(ul, u, l):
	pred = np.zeros((4,4))
	pred[0,3] = u[1]//4 + u[2]//2 + u[3]//4
	pred[0,2] = u[0]//4 + u[1]//2 + u[2]//4
	pred[1,3] = pred[0,2]
	pred[0,1] = ul//4 + u[0]//2 + u[1]//4
	pred[1,2] = pred[0,1]
	pred[2,3] = pred[0,1]
	pred[0,0] = ul//4 + u[0]//2 + l[0]//4
	pred[1,1] = pred[0,0]
	pred[2,2] = pred[0,0]
	pred[3,3] = pred[0,0]
	pred[1,0] = u[0]//4 + l[0]//2 + l[1]//4
	pred[2,1] = pred[1,0]
	pred[3,2] = pred[1,0]
	pred[2,0] = l[0]//4 + l[1]//2 + l[2]//4
	pred[3,1] = pred[2,0]
	pred[3,0] = l[1]//4 + l[2]//2 + l[3]//4
	return pred

def verticalright4x4(ul, u, l):
	pred = np.zeros((4,4))
	pred[0,0] = ul//2 + u[0]//2
	pred[2,1] = pred[0,0]
	pred[0,1] = u[0]//2 + u[1]//2
	pred[2,2] = pred[0,1]
	pred[0,2] = u[1]//2 + u[2]//2
	pred[2,3] = pred[0,2]
	pred[0,3] = u[2]//2 + u[3]//2
	pred[1,0] = u[0]//4 + ul//2 + l[0]//4
	pred[3,1] = pred[1,0]
	pred[1,1] = ul//4 + u[0]//2 + u[1]//4
	pred[3,2] = pred[1,1]
	pred[1,2] = u[0]//4 + u[1]//2 + u[2]//4
	pred[3,3] = pred[1,2]
	pred[1,3] = u[1]//4 + u[2]//2 + u[3]//4
	pred[2,0] = ul//4 + l[0]//2 + l[1]//4
	pred[3,0] = l[0]//4 + l[1]//2 + l[2]//4
	return pred

def horizontaldown4x4(ul, u, l):
	pred = np.zeros((4,4))
	pred[0,0] = ul//2 + l[0]//2
	pred[1,2] = pred[0,0]
	pred[0,1] = u[0]//4 + ul//2 + l[0]//4
	pred[1,3] = pred[0,1]
	pred[0,2] = ul//4 + u[0]//2 + u[1]//4
	pred[0,3] = u[0]//4 + u[1]//2 + u[2]//4
	pred[1,0] = l[0]//2 + l[1]//2
	pred[2,2] = pred[1,0]
	pred[1,1] = ul//4 + l[1]//2 + l[2]//4
	pred[2,3] = pred[1,1]
	pred[2,0] = l[1]//2 + l[2]//2
	pred[3,2] = pred[2,0]
	pred[2,1] = l[0]//4 + l[1]//2 + l[2]//4
	pred[3,3] = pred[2,1]
	pred[3,0] = l[2]//2 + l[3]//2
	pred[3,1] = l[1]//4 + l[2]//2 + l[3]//4
	return pred

def verticalleft4x4(u, ur):
	pred = np.zeros((4,4))
	pred[0,0] = u[0]//2 + u[1]//2
	pred[0,1] = u[1]//2 + u[2]//2
	pred[2,0] = pred[0,1]
	pred[0,2] = u[2]//2 + u[3]//2
	pred[2,1] = pred[0,2]
	pred[0,3] = u[3]//2 + ur[0]//2
	pred[2,2] = pred[0,3]
	pred[2,3] = ur[0]//2 + ur[1]//2
	pred[1,0] = u[0]//4 + u[1]//2 + u[2]//4
	pred[1,1] = u[1]//4 + u[2]//2 + u[3]//4
	pred[3,0] = pred[1,1]
	pred[1,2] = u[2]//4 + u[3]//2 + ur[0]//4
	pred[3,1] = pred[1,2]
	pred[1,3] = u[3]//4 + ur[0]//2 + ur[1]//4
	pred[3,2] = pred[1,3]
	pred[3,3] = ur[0]//4 + ur[1]//2 + ur[2]//4
	return pred

def horizontalup4x4(l):
	pred = np.zeros((4,4))
	pred[0,0] = l[0]//2 + l[1]//2
	pred[0,1] = l[0]//4 + l[1]//2 + l[2]//4
	pred[0,2] = l[1]//2 + l[2]//2
	pred[1,0] = pred[0,2]
	pred[0,3] = l[1]//4 + l[2]//2 + l[3]//4
	pred[1,1] = pred[0,3]
	pred[1,2] = l[2]//2 + l[3]//2
	pred[2,0] = pred[1,2]
	pred[1,3] = l[2]//4 + 3*l[3]//4
	pred[2,1] = pred[1,3]
	pred[3,0] = l[3]
	pred[2,2] = l[3]
	pred[2,3] = l[3]
	pred[3,1] = l[3]
	pred[3,2] = l[3]
	pred[3,3] = l[3]
	return pred

def vertical16x16(u):
	pred = np.zeros((16,16))
	for i in range(16):
		pred[i] = u
	return pred

def horizontal16x16(l):
	pred = np.zeros((16,16))
	for i in range(16):
		pred[:,i] = l
	return pred

def dc16x16(u, l):
	pred = np.ones((16,16))
	avg = (np.sum(u) + np.sum(l)) // 32
	pred = pred * avg
	return pred

def vertical8x8(u):
	pred = np.zeros((8,8))
	for i in range(8):
		pred[i] = u
	return pred

def horizontal8x8(l):
	pred = np.zeros((8,8))
	for i in range(8):
		pred[:,i] = l
	return pred

def dc8x8(u, l):
	pred = np.ones((8,8))
	avg = (np.sum(u) + np.sum(l)) // 16
	pred = pred * avg
	return pred
