import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DRAW_MBLOCK_INDEX = False

BLOCK_SIZE = 16
img1 = cv2.imread('../../images/corgi-underwater/12.jpg')
img2 = cv2.imread('../../images/corgi-underwater/16.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
vheight, vwidth, channels = img1.shape
imshape = img1.shape

fig, ((ax_ref, ax_input), (ax_match_block, ax_block)) = plt.subplots(
    2, 2, figsize=(12, 8))

# MOTION ESTIMATION
# We perform full search over the reference frame


def find_match(img, block):
    global BLOCK_SIZE
    image = img
    best_match = 9999999999
    best_coord = [0, 0]
    best_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
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
                best_block = ref_block
    print("Best cood", best_coord[1], best_coord[0])

    # plot reference frame
    ax_ref.imshow(img)
    ax_ref.set_title('Reference Frame')

    # plot template (macroblock we are searching for)
    ax_block.imshow(block)
    ax_block.set_title('Block Searching For Match')

    # plot match coordinate
    ax_ref.add_patch(Rectangle((best_coord[1], best_coord[0]), BLOCK_SIZE,
                               BLOCK_SIZE,  edgecolor='red', fill=False, lw=2))
    ax_ref.text(best_coord[1] + 10, best_coord[0] +
                25, "BEST MATCH: (" + str(best_coord[1]) + "," + str(best_coord[0]) + ")", fontsize=8)

    return [best_coord, best_block]


def split_frame_into_mblocks(input_frame, highlightBlock):
    global BLOCK_SIZE
    blocks = []

    # iterate over blocks
    ax_input.imshow(input_frame)
    ax_input.set_title("Input Frame")
    block_idx = 0
    for i in range(0, imshape[0], BLOCK_SIZE):
        if (i+BLOCK_SIZE > imshape[0]):
            continue
        for j in range(0, imshape[1], BLOCK_SIZE):
            if (j+BLOCK_SIZE > imshape[1]):
                continue
            block = input_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            # draw out bounding box for each block
            if DRAW_MBLOCK_INDEX:
                ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                             BLOCK_SIZE,  edgecolor='black', fill=False, lw=0.5))
                ax_input.text(j, i, block_idx, fontsize=3)
            blocks.append(block)

            if (block_idx == highlightBlock):
                ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                             BLOCK_SIZE,  edgecolor='red', fill=False, lw=2))
                ax_input.text(j + 10, i + 25,  "SEARCH BLOCK: (" +
                              str(j) + "," + str(i) + ")",  fontsize=8)
                highlightCoord = [i, j]
            block_idx += 1

    print("Finished splitting frame into macro blocks")
    return [blocks, highlightCoord]


TEST_BLOCK = 120
[blocks, highlightCoord] = split_frame_into_mblocks(
    img2, highlightBlock=TEST_BLOCK)
print("Blocks", len(blocks))
[best_coord, best_block] = find_match(img1, blocks[TEST_BLOCK])
ax_match_block.imshow(best_block)
ax_match_block.set_title("Best Match Block in Ref Frame")
dx = best_coord[1] - highlightCoord[1]
dy = best_coord[0] - highlightCoord[0]
print("deltas", dx, dy)
ax_input.arrow(highlightCoord[1], highlightCoord[0] + 8, dx, dy, head_width=10)

plt.show()
