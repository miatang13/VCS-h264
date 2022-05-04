import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

DRAW_MBLOCK_INDEX = False
DRAW_SEARCH_WINDOW_BLOCKS = True

# Tweak params
BLOCK_SIZE = 16
SEARCH_WINDOW_SIZE = 3
FONT_SIZE = 5

# We just grab two images for testing
img1 = cv2.imread('../../images/oscar-cat/16.jpg')
img2 = cv2.imread('../../images/oscar-cat/17.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
vheight, vwidth, channels = img1.shape
imshape = img1.shape

fig, ((ax_ref, ax_input), (ax_match_block, ax_block)) = plt.subplots(
    2, 2, figsize=(12, 8))

# MOTION ESTIMATION
# We perform full search over the reference frame


def find_match(img, block, block_coord):
    print("Find match input block shape", block.shape)

    global BLOCK_SIZE
    image = img
    best_match = 9999999999
    best_coord = [0, 0]
    best_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

    # Loop over all blocks in reference frame within the search window
    dist_from_block = BLOCK_SIZE * SEARCH_WINDOW_SIZE
    # Mins
    i_min = max(block_coord[1] - dist_from_block, 0)
    j_min = max(block_coord[0] - dist_from_block, 0)
    # Max
    i_max = min(block_coord[1] + dist_from_block, imshape[0])
    j_max = min(block_coord[0] + dist_from_block, imshape[1])

    print("Search window", i_min, i_max, ",", j_min, j_max)

    # Search for the block
    for i in range(i_min, i_max, BLOCK_SIZE):

        # Check if out of bound of the image
        if (i+BLOCK_SIZE > imshape[0]):
            continue
        for j in range(j_min, j_max, BLOCK_SIZE):
            if (j+BLOCK_SIZE > imshape[1]):
                continue

            if (DRAW_SEARCH_WINDOW_BLOCKS):
                ax_ref.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                           BLOCK_SIZE,  edgecolor='gray', fill=False, lw=0.5))
            # Valid block within image bound
            ref_block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            # print("Ref block size", ref_block.shape)

            # Get sum difference between the 2 blocks
            diff = np.sum(np.abs(ref_block - block))

            # Update if it is a better block
            if (diff < best_match):
                best_match = diff
                best_coord = [j, i]
                best_block = ref_block
    print("Best coord", best_coord[1], best_coord[0])

    # plot reference frame
    ax_ref.imshow(img)
    ax_ref.set_title('Reference Frame')

    # plot template (macroblock we are searching for)
    ax_block.imshow(block)
    ax_block.set_title('Block Searching For Match')

    # plot search window
    ax_ref.add_patch(Rectangle((j_min, i_min), j_max - j_min, i_max - i_min,
                     edgecolor='orange', fill=False, lw=2))
    ax_ref.text(j_max - BLOCK_SIZE, i_max +
                BLOCK_SIZE, "SEARCH WINDOW", fontsize=FONT_SIZE, color="black")

    # plot search window center
    ax_ref.add_patch(Rectangle((block_coord[0], block_coord[1]), BLOCK_SIZE, BLOCK_SIZE,
                               edgecolor='yellow', fill=False, lw=2))
    ax_ref.text(block_coord[0] - BLOCK_SIZE, block_coord[1] +
                BLOCK_SIZE * 2, "CENTER", fontsize=FONT_SIZE, color="white")

    # plot match coordinate
    ax_ref.add_patch(Rectangle((best_coord[0], best_coord[1]), BLOCK_SIZE,
                               BLOCK_SIZE,  edgecolor='red', fill=False, lw=2))
    ax_ref.text(best_coord[0] + BLOCK_SIZE, best_coord[1] +
                BLOCK_SIZE * 2, "BEST MATCH: (" + str(best_coord[0]) + "," + str(best_coord[1]) + ")", fontsize=FONT_SIZE, color="white")

    # plot match block
    ax_match_block.imshow(best_block)
    ax_match_block.set_title("Best Match Block in Ref Frame")
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
                ax_input.text(j, i + BLOCK_SIZE, block_idx,
                              fontsize=3, color="white")

            # we keep all blocks
            blocks.append(block)

            # draw the block we are searching for
            if (block_idx == highlightBlock):
                ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                             BLOCK_SIZE,  edgecolor='yellow', fill=False, lw=2))
                ax_input.text(j + 10, i + BLOCK_SIZE * 2,  "SEARCH BLOCK: (" +
                              str(j) + "," + str(i) + ")",  fontsize=FONT_SIZE, color="white")
                highlightCoord = [j, i]

            # we use this index to decide on which block to highlight
            block_idx += 1

    print("Finished splitting frame into macro blocks")
    return [blocks, highlightCoord]


def get_motion_vector(match_coord, search_coord):
    # We want vector from search coordinate to match coordinate
    dx = match_coord[0] - search_coord[0]
    dy = match_coord[1] - search_coord[1]
    print("deltas", dx, dy)
    motion_vector = [dx, dy]
    return motion_vector


TEST_BLOCK_IDX = 945
[blocks, searchCoord] = split_frame_into_mblocks(img2, TEST_BLOCK_IDX)
print("Num Blocks", len(blocks))
[best_coord, best_block] = find_match(
    img1, blocks[TEST_BLOCK_IDX], searchCoord)
motion_vector = get_motion_vector(best_coord, searchCoord)
ax_input.arrow(searchCoord[0] + BLOCK_SIZE/2, searchCoord[1] + BLOCK_SIZE/2,
               motion_vector[0], motion_vector[1], head_width=10, edgecolor="yellow")

plt.show()
