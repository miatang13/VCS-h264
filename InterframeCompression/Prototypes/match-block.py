import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import scipy.signal as sp
import math

BLOCK_SIZE = 16
img1 = cv2.imread('../../images/corgi-underwater/12.jpg')
img2 = cv2.imread('../../images/corgi-underwater/16.jpg')
vheight, vwidth, channels = img1.shape
imshape = img1.shape


fig, (ax_ref, ax_input, ax_block) = plt.subplots(1, 3, figsize=(15, 6))


# MOTION ESTIMATION
# We perform full search over the reference frame


def find_match(img, block):
    image = img
    best_match = 9999999999
    best_coord = [0, 0]
    for i in range(0, imshape[0], BLOCK_SIZE):
        if (i+BLOCK_SIZE > imshape[0]):
            continue
        for j in range(0, imshape[1], BLOCK_SIZE):
            if (j+BLOCK_SIZE > imshape[1]):
                continue
            ref_block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            diff = np.sum(np.abs(ref_block - block))
            if (diff < best_match):
                best_match = diff
                best_coord = [i, j]
    print("Best cood", best_coord)

    # plot reference frame
    ax_ref.imshow(img, cmap='gray')
    ax_ref.set_title('Reference Frame')
    ax_ref.set_axis_off()

    # plot template (macroblock we are searching for)
    ax_block.imshow(block, cmap='gray')
    ax_block.set_title('Block')
    ax_block.set_axis_off()

    # plot match coordinate
    ax_ref.add_patch(Rectangle((best_coord[1], best_coord[0]), BLOCK_SIZE,
                               BLOCK_SIZE,  edgecolor='red', fill=False, lw=2))
    ax_ref.text(best_coord[1] + 10, best_coord[0] +
                25, "BEST MATCH: (" + str(best_coord[1]) + "," + str(best_coord[0]) + ")", fontsize=8)

    return best_coord


def split_frame_into_mblocks(input_frame, highlightBlock):
    blocks = []

    # iterate over blocks
    ax_input.imshow(input_frame, cmap="gray")
    ax_input.set_title("Input Frame")
    ax_block.set_axis_off()
    block_idx = 0
    for i in range(0, imshape[0], BLOCK_SIZE):
        if (i+BLOCK_SIZE > imshape[0]):
            continue
        for j in range(0, imshape[1], BLOCK_SIZE):
            if (j+BLOCK_SIZE > imshape[1]):
                continue
            block = input_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            # draw out bounding box for each block
            # ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
            #                              BLOCK_SIZE,  edgecolor='black', fill=False, lw=0.5))
            # ax_input.text(j, i, block_idx, fontsize=3)
            blocks.append(block)

            if (block_idx == highlightBlock):
                ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                             BLOCK_SIZE,  edgecolor='red', fill=False, lw=3))
                ax_input.text(j + 10, i + 25,  "SEARCH BLOCK: (" +
                              str(j) + "," + str(i) + ")",  fontsize=8)
            block_idx += 1

    print("Finished splitting frame into macro blocks")
    return blocks


TEST_BLOCK = 380
blocks = split_frame_into_mblocks(img2, highlightBlock=TEST_BLOCK)
print("Blocks", len(blocks))
find_match(img1, blocks[TEST_BLOCK])

plt.show()
