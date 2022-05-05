import numpy as np
import math
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

TEST_COMPRESSOR = False

# jpeg standard quantization matrices for lum and chrom
QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 48, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])

QF = 50.0  # quality factor, must be between 1 and 99
if QF < 50 and QF > 1:
    scale = 50/QF
elif QF < 100:
    scale = (100-QF)/50
else:
    print("Invalid quality setting, must be between 1 and 99.")
Q = [np.clip(np.round(QY*scale), 1, 255),
     np.clip(np.round(QC*scale), 1, 255),
     np.clip(np.round(QC*scale), 1, 255)]


class DCTCompressor:

    def __init__(self, block_size):
        self.blocksize = block_size
        self.compressed = []
        self.Q = Q

    # These are the same as the completeDCT steps, just broken down into 2 functions
    def compress(self, input_bgrimg):
        imshape = input_bgrimg.shape
        # resize to closest multiple of block for convenience
        bgrimg = cv2.resize(
            input_bgrimg, (self.blocksize*(imshape[1]//self.blocksize), self.blocksize*(imshape[0]//self.blocksize)))
        imshape = bgrimg.shape
        YCrCbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(YCrCbimg)
        Y = np.array(Y).astype(np.int16) - 128
        Cr = np.array(Cr).astype(np.int16) - 128
        Cb = np.array(Cb).astype(np.int16) - 128
        channels = [Y, Cr, Cb]
        print("begin compression")
        compressed = []
        for channel in range(3):
            image = channels[channel]
            result = np.zeros(image.shape)
            for i in tqdm(range(0, imshape[0], self.blocksize), leave=False):
                for j in range(0, imshape[1], self.blocksize):
                    block = image[i:i+self.blocksize, j:j+self.blocksize]
                    d = self._dct2(block)  # dct(block) #perform transform
                    # perform quantization
                    d = np.round(np.divide(d, self.Q[channel]))
                    result[i:i+self.blocksize, j:j+self.blocksize] = d
            compressed.append(result)
        return compressed

    def decompress(self, compressed, imshape):
        print("begin decompression")
        decompressed = []
        for channel in range(3):
            image = compressed[channel]
            result = np.zeros(image.shape)
            for i in tqdm(range(0, imshape[0], self.blocksize)):
                for j in range(0, imshape[1], self.blocksize):
                    block = image[i:i+self.blocksize, j:j+self.blocksize]
                    # perform de-quantization
                    d = np.multiply(block, self.Q[channel])
                    d = self._idct2(d)
                    result[i:i+self.blocksize, j:j+self.blocksize] = d
            decompressed.append(result.astype(np.uint8)+128)
        print("decompression finished")
        newYCrCb = np.dstack(decompressed)
        newBGR = cv2.cvtColor(newYCrCb, cv2.COLOR_YCR_CB2BGR).astype(np.uint8)
        return newBGR

    #################################################################
    # PRIVATE METHODS

    # TESTING
    # used to pass in image to process 1 channel
    def _completeDCT(self, input_img):
        imshape = (self.blocksize*input_img.shape[0]//self.blocksize, self.blocksize*input_img.shape[1]//self.blocksize, 3)
        print("begin compression")
        compressed = self.compress(input_img)
        newBGR = self.decompress(compressed, imshape)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title("Compressed Image")
        plt.imshow(cv2.cvtColor(newBGR, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    def _dct2(self, matrix):
        dctmat = self._dctMatrix()
        result = np.matmul(dctmat, matrix)
        result = np.matmul(result, dctmat.transpose())
        return result

    def _idct2(self, matrix):
        dctmat = self._dctMatrix()
        result = np.matmul(dctmat.transpose(), matrix)
        result = np.matmul(result, dctmat)
        return result

    # 8x8 matrix format
    def _dctMatrix(self):
        result = np.zeros((self.blocksize,  self.blocksize))
        for i in range(self.blocksize):
            for j in range(self.blocksize):
                if i == 0:
                    result[i, j] = 1/math.sqrt(self.blocksize)
                else:
                    result[i, j] = math.sqrt(2 / self.blocksize) * \
                        math.cos((2*j+1)*i*math.pi/(2 * self.blocksize))
        return result

    def _cuHelper(self, ind):
        if ind == 0:
            return 1/math.sqrt(2)
        else:
            return 1


if (TEST_COMPRESSOR):
    blocksize = 8

    # Prepare plots
    bgrimg = cv2.imread('../images/screm.png')
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Original Image")
    plt.imshow(cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Construct compressor
    c = DCTCompressor(blocksize)
    c._completeDCT(bgrimg)

    # Done
    plt.show()
    print("Finished!")
