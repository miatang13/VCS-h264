import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DRAW_MBLOCK_INDEX = False
DRAW_SEARCH_WINDOW_BLOCKS = False
DRAW_SEARCH_BLOCK = False

# Tweak params
BLOCK_SIZE = 16
SEARCH_WINDOW_SIZE = 2
FONT_SIZE = 5
LINE_WIDTH = 0.5
SIMILARITY_THRESHOLD = 255 * .1

# We just grab two images for testing
img1 = cv2.imread('../../images/sequences/minor-jump/0.png')
img2 = cv2.imread('../../images/sequences/minor-jump/1.png')
overlayImg = cv2.imread("../../images/sequences/minor-jump/overlay.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
overlayImg = cv2.cvtColor(overlayImg, cv2.COLOR_BGR2RGB)
vheight, vwidth, channels = img1.shape
imshape = img1.shape

block_coords = []

fig, ((ax_ref_full, ax_overlay), (ax_ref, ax_input), (ax_match_block, ax_block)) = plt.subplots(
    3, 2, figsize=(10.5, 10.5))

ax_overlay.imshow(overlayImg)
ax_overlay.set_title("Overlay frames")


# MOTION ESTIMATION
# We perform full search over the reference frame


def find_match(img, block, block_coord):
    global BLOCK_SIZE
    # Returning these
    best_coord = [0, 0]
    best_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

    # For plots
    # plot reference frame
    ax_ref.imshow(img)
    ax_ref.set_title('Reference Frame annot. w/ nonstatic blocks')
    ax_ref_full.imshow(img)
    ax_ref_full.set_title('Reference Frame')

    # OPTIMIZE SEARCH
    # return input location if the same location on reference frame is very similar since motion prediction is
    # based on the assumption that there are many static pixels across frames
    # evaluate input block coord on ref frame
    block_at_input_location = img[block_coord[1]: block_coord[1] +
                                  BLOCK_SIZE, block_coord[0]:block_coord[0]+BLOCK_SIZE]
    # np.sum(np.abs(block_at_input_location - block))
    diff = np.sum(np.abs(cv2.subtract(block_at_input_location, block)))
    print("diff", diff)
    if (diff <= 1000):
        print("Input position is good. Block is static. ")
        best_coord = block_coord
        best_block = block_at_input_location
    else:
        # print("Input position is not static. Searching.")
        # The block isn't static, we start searching
        image = img
        best_match = 9999999999

        # Loop over all blocks in reference frame within the search window
        dist_from_block = round(BLOCK_SIZE * SEARCH_WINDOW_SIZE)
        # Mins
        i_min = max(block_coord[1] - dist_from_block, 0)
        j_min = max(block_coord[0] - dist_from_block, 0)
        # Max
        i_max = min(block_coord[1] + dist_from_block, imshape[0])
        j_max = min(block_coord[0] + dist_from_block, imshape[1])

        # print("Search window", i_min, i_max, ",", j_min, j_max)

        # Search for the block
        STEP_SIZE = round(BLOCK_SIZE/3)
        for i in range(i_min, i_max, STEP_SIZE):

            # Check if out of bound of the image
            if (i+BLOCK_SIZE > i_max):
                continue
            for j in range(j_min, j_max, STEP_SIZE):
                if (j+BLOCK_SIZE > j_max):
                    continue

                if (DRAW_SEARCH_WINDOW_BLOCKS):
                    ax_ref.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                               BLOCK_SIZE,  edgecolor='gray', fill=False, lw=LINE_WIDTH))
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
        # print("Best coord", best_coord[1], best_coord[0])
        # plot search window
        # ax_ref.add_patch(Rectangle((j_min, i_min), j_max - j_min, i_max - i_min,
        #                            edgecolor='orange', fill=False, lw=LINE_WIDTH))
        # ax_ref.text(j_max - BLOCK_SIZE, i_max +
        #             BLOCK_SIZE, "SEARCH WINDOW", fontsize=FONT_SIZE, color="black", backgroundcolor="white")

        # plot search window center
        ax_ref.add_patch(Rectangle((block_coord[0], block_coord[1]), BLOCK_SIZE, BLOCK_SIZE,
                                   edgecolor='yellow', fill=False, lw=LINE_WIDTH))
        # ax_ref.text(block_coord[0] - BLOCK_SIZE, block_coord[1] +
        #             BLOCK_SIZE * 2, "CENTER", fontsize=FONT_SIZE, color="white")
        # plot match coordinate
        # ax_ref.add_patch(Rectangle((best_coord[0], best_coord[1]), BLOCK_SIZE,
        #                            BLOCK_SIZE,  edgecolor='red', fill=False, lw=LINE_WIDTH * 2))

    # plot template (macroblock we are searching for)
    ax_block.imshow(block)
    ax_block.set_title('Block Searching For Match')

    # ax_ref.text(best_coord[0] + BLOCK_SIZE, best_coord[1] +
    #             BLOCK_SIZE * 4, "BEST MATCH", fontsize=FONT_SIZE, color="black", backgroundcolor="white")

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
                                             BLOCK_SIZE,  edgecolor='black', fill=False, lw=LINE_WIDTH))
                ax_input.text(j, i + BLOCK_SIZE, block_idx,
                              fontsize=3, color="white")

            # we keep all blocks
            blocks.append(block)

            # draw the block we are searching for
            if (DRAW_SEARCH_BLOCK and block_idx == highlightBlock):
                ax_input.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                             BLOCK_SIZE,  edgecolor='yellow', fill=False, lw=LINE_WIDTH))
                ax_input.text(j + BLOCK_SIZE*2, i + BLOCK_SIZE * 2,  "SEARCH BLOCK",
                              fontsize=FONT_SIZE, color="black", backgroundcolor="white")
                highlightCoord = [j, i]

            # we use this index to decide on which block to highlight
            block_idx += 1
            block_coords.append([j, i])

    print("Finished splitting frame into macro blocks")
    return [blocks, None]


def get_motion_vector(match_coord, search_coord):
    # We want vector from search coordinate to match coordinate
    dx = match_coord[0] - search_coord[0]
    dy = match_coord[1] - search_coord[1]
    # print("deltas", dx, dy)
    motion_vector = [dx, dy]
    return motion_vector


TEST_BLOCK_IDX = 298
[blocks, _] = split_frame_into_mblocks(img1, TEST_BLOCK_IDX)
print("Num Blocks", len(blocks))
for block_i in range(len(blocks)):
    searchCoord = block_coords[block_i]
    [best_coord, best_block] = find_match(
        img2, blocks[block_i], searchCoord)
    motion_vector = get_motion_vector(best_coord, searchCoord)
    if (not (motion_vector[0] == 0 and motion_vector[1] == 0)):
        # Only plot vectors when the block is not static
        ax_overlay.arrow(searchCoord[0] + BLOCK_SIZE/2, searchCoord[1] + BLOCK_SIZE/2,
                         motion_vector[0], motion_vector[1], head_width=5, edgecolor="yellow")
    print("Block index:", block_i)


plt.show()

print("Finished!")
