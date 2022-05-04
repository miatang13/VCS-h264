import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DRAW_MBLOCK_INDEX = False
DRAW_SEARCH_WINDOW_BLOCKS = False
DRAW_SEARCH_BLOCK = False

# Different inputs
TEST_CORGI = False
TEST_SIMPLE = True

# Tweak params
FONT_SIZE = 5
PLOT_FONT_SIZE = 12
LINE_WIDTH = 0.5
SIMILARITY_THRESHOLD = 2000

# We just grab two images for testing
img1 = cv2.imread('../../images/oscar-cat/5.jpg')
img2 = cv2.imread('../../images/oscar-cat/6.jpg')
overlayImg = cv2.imread('../../images/oscar-cat/5+6.jpg')

if TEST_CORGI:
    img1 = cv2.imread("../../images/corgi-underwater/15.jpg")
    img2 = cv2.imread("../../images/corgi-underwater/16.jpg")
    overlayImg = cv2.imread("../../images/corgi-underwater/16.jpg")
elif TEST_SIMPLE:
    img1 = cv2.imread("../../images/sequences/minor-jump/0.png")
    img2 = cv2.imread("../../images/sequences/minor-jump/1.png")
    overlayImg = cv2.imread("../../images/sequences/minor-jump/overlay.png")

refImage = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
inputImage = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
overlayImg = cv2.cvtColor(overlayImg, cv2.COLOR_BGR2RGB)
vheight, vwidth, channels = refImage.shape
imshape = inputImage.shape

# We want smaller blocks for smaller images to avoid blocky reconstruction
BLOCK_SIZE = 16  # 4 if imshape[0] * imshape[1] <= 150000 else 16
print("Block size: ", BLOCK_SIZE)
SEARCH_WINDOW_SIZE = 2 * BLOCK_SIZE

fig, ((ax_ref, ax_input, ax_overlay), (ax_replaced, ax_residual, ax_reconstructed)) = plt.subplots(
    2, 3, figsize=(14, 9))
fig.suptitle("Motion Prediction & Compensation", fontsize=16)

ax_overlay.imshow(overlayImg)
ax_overlay.set_title("Overlay Frames with Highlighted Nonstatic Blocks")


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
    ax_ref.set_title('Reference Frame')

    # OPTIMIZE SEARCH
    # return input location if the same location on reference frame is very similar since motion prediction is
    # based on the assumption that there are many static pixels across frames
    # evaluate input block coord on ref frame
    block_at_input_location = img[block_coord[1]: block_coord[1] +
                                  BLOCK_SIZE, block_coord[0]:block_coord[0]+BLOCK_SIZE]
    # np.sum(np.abs(block_at_input_location - block))
    diff = np.sum(np.abs(cv2.subtract(block_at_input_location, block)))
    if (diff <= SIMILARITY_THRESHOLD):
        # print("Input position is good. Block is static. ")
        best_coord = block_coord
        best_block = block_at_input_location
    else:
        # print("Input position is not static. Searching.")
        # The block isn't static, we start searching
        image = img
        best_match = 9999999999

        # Loop over all blocks in reference frame within the search window
        dist_from_block = SEARCH_WINDOW_SIZE
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
            if (i+BLOCK_SIZE >= i_max):
                continue
            for j in range(j_min, j_max, STEP_SIZE):
                if (j+BLOCK_SIZE >= j_max):
                    continue

                if (DRAW_SEARCH_WINDOW_BLOCKS):
                    ax_ref.add_patch(Rectangle((j, i), BLOCK_SIZE,
                                               BLOCK_SIZE,  edgecolor='gray', fill=False, lw=LINE_WIDTH))
                # Valid block within image bound
                ref_block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                if (ref_block.shape[0] != ref_block.shape[1]):
                    print("Ref block has invalid shape", ref_block.shape)
                # Get sum difference between the 2 blocks
                diff = np.sum(np.abs(ref_block - block))

                # Update if it is a better block
                if (diff < best_match):
                    best_match = diff
                    best_coord = [j, i]
                    best_block = ref_block

        # plot search window center
        ax_overlay.add_patch(Rectangle((block_coord[0], block_coord[1]), BLOCK_SIZE, BLOCK_SIZE,
                                       edgecolor='yellow', fill=False, lw=LINE_WIDTH * 2))
    return [best_coord, best_block]


def split_frame_into_mblocks(input_frame):
    global BLOCK_SIZE
    blocks = []
    block_coords = []

    # iterate over blocks
    ax_input.imshow(input_frame)
    ax_input.set_title("Input Frame")
    block_idx = 0
    for i in range(0, imshape[0], BLOCK_SIZE):
        if (i+BLOCK_SIZE > imshape[0]):
            break
        for j in range(0, imshape[1], BLOCK_SIZE):
            if (j+BLOCK_SIZE > imshape[1]):
                break

            block = input_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

            # we keep all blocks
            blocks.append(block)

            # we keep track of coordinates for later
            block_idx += 1
            block_coords.append([j, i])

    print("Finished splitting frame into macro blocks")
    return [blocks, block_coords]


def get_motion_vector(match_coord, search_coord):
    # We want vector from search coordinate to match coordinate
    dx = match_coord[0] - search_coord[0]
    dy = match_coord[1] - search_coord[1]
    # print("deltas", dx, dy)
    motion_vector = [dx, dy]
    return motion_vector


def process_motion_prediction(input_frame, ref_frame):
    # Break input frame into 16x16 blocks
    [blocks, block_coords] = split_frame_into_mblocks(input_frame)
    print("Num Blocks", len(blocks))

    # For each block, we get its motoin vector to ref frame (if it is not static)
    motion_vectors = []
    for block_i in range(len(blocks)):
        searchCoord = block_coords[block_i]
        # print(searchCoord)

        # We find the matching block in reference frame
        [best_coord, best_block] = find_match(
            ref_frame, blocks[block_i], searchCoord)
        motion_vector = get_motion_vector(best_coord, searchCoord)

        motion_vectors.append(motion_vector)

        # Only care about vectors when the block is not static
        if (not (motion_vector[0] == 0 and motion_vector[1] == 0)):
            ax_overlay.arrow(searchCoord[0] + BLOCK_SIZE/2, searchCoord[1] + BLOCK_SIZE/2,
                             motion_vector[0], motion_vector[1], head_width=5, edgecolor="yellow")
        # print("Finished processing block index:", block_i)

    return [motion_vectors, block_coords]


def reconstruct_from_motion_vectors(motion_vectors, ref_frame, block_coords):
    # We reconstruct an image solely from the motion vectors and the reference frame
    # which we want to compare with the actual input frame to get the residual frame
    reconstruct_img = np.zeros(
        (imshape[0], imshape[1], channels), ref_frame.dtype)
    num_static = 0
    for block_i in range(len(block_coords)):
        print("Reconstruct block", block_i)
        [j, i] = block_coords[block_i]
        v = motion_vectors[block_i]
        sampled_block = np.zeros(
            (BLOCK_SIZE, BLOCK_SIZE, channels), ref_frame.dtype)
        if (v[0] == 0 and v[1] == 0):
            # We find static blocks and directly sample from the reference frame
            sampled_block = ref_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            num_static += 1
        else:
            # For nonstatic blocks, we find the motion predicted block
            i_0 = i + v[1]
            j_0 = j + v[0]
            sampled_block = ref_frame[i_0:i_0 +
                                      BLOCK_SIZE, j_0:j_0+BLOCK_SIZE]
        reconstruct_img[i:i+BLOCK_SIZE, j:j +
                        BLOCK_SIZE] = sampled_block
        print("Finished reconstructing block", block_i)

    ax_replaced.imshow(reconstruct_img)
    ax_replaced.set_title("Reconstructed From Motion Vectors")

    print("There are", num_static, "static blocks out of",
          len(block_coords), "blocks")
    return reconstruct_img


def get_residuals(input_frame, reconstructed):
    delta = input_frame - reconstructed
    ax_residual.imshow(delta)
    ax_residual.set_title("Residuals")
    return delta


def fully_reconstruct(residuals, img):
    result = img + residuals
    ax_reconstructed.imshow(result)
    ax_reconstructed.set_title("Final Result")
    return result


# Pipeline
# 1. Get motion vectors for each block
[motion_vecs, coords] = process_motion_prediction(
    input_frame=inputImage, ref_frame=refImage)
# 2. Reconstruct image from reference img and motion vectors
reconstructed_img = reconstruct_from_motion_vectors(
    motion_vectors=motion_vecs, ref_frame=refImage, block_coords=coords)
# 3. Get residuals for better result
residuals = get_residuals(input_frame=inputImage,
                          reconstructed=reconstructed_img)
# 4. Results
reconstruct_w_res = fully_reconstruct(residuals, reconstructed_img)

plt.show()

print("Finished!")
