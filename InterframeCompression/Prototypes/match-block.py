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

    return best_coord


def split_frame_into_mblocks(input_frame, highlightBlock):
    blocks = []

    # iterate over blocks
    # for col in range(0, num_blocks_hor):
    #     start_col = col * BLOCK_SIZE
    #     for row in range(0, num_blocks_vert):
    #         start_row = row * BLOCK_SIZE
    #         # (start_i, start_j) is the left upper corner of each block
    #         for i in range(0, BLOCK_SIZE):
    #             if (start_col + i >= vwidth):
    #                 continue
    #             for j in range(0, BLOCK_SIZE):
    #                 if (start_row + j >= vheight):
    #                     continue
    #                 # we collect each entry in the block
    #                 # print("input x, y", start_col + i, start_row + j)
    #                 blocks[col][row][j][i] = input_frame[start_row +
    #                                                      j][start_col + i]
    ax_input.imshow(input_frame, cmap="gray")
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
            block_idx += 1

    print("Finished splitting frame into macro blocks")
    return blocks


TEST_BLOCK = 10
blocks = split_frame_into_mblocks(img2, highlightBlock=TEST_BLOCK)
print("Blocks", len(blocks))
find_match(img1, blocks[TEST_BLOCK])

plt.show()
