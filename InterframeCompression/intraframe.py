import numpy as np

# Operations on individual frames


def cuHelper(ind):
    if ind == 0:
        return 1/math.sqrt(2)
    else:
        return 1

# reference: https://www.math.cuhk.edu.hk/~lmlui/dct.pdf


def dct(inputBlock):
    result = np.zeros((blockh, blockw))
    for i in range(blockh):
        for j in range(blockw):
            cosSum = 0
            for k in range(blockh):
                for l in range(blockw):
                    temp = inputBlock[k][l] * math.cos((2*k+1)*i*math.pi/(
                        2*blockh)) * math.cos((2*l+1)*j*math.pi/(2*blockw))
                    cosSum += temp
            result[i][j] = (4/(blockh*blockw)) * \
                cuHelper(i) * cuHelper(j) * cosSum
    return result
