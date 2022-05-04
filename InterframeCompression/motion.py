import numpy as np
import cv2

# MOTION ESTIMATION

# Optimization Param
SIMILARITY_THRESHOLD = 2000
CHANNELS = 3

# We want smaller blocks for smaller images to avoid blocky reconstruction
BLOCK_SIZE = 16  # 4 if vidshape[0] * vidshape[1] <= 150000 else 16
print("Block size: ", BLOCK_SIZE)
SEARCH_WINDOW_SIZE = 2 * BLOCK_SIZE


class MotionProcessor:
    def __init__(self, block_size):
        self.block_size = block_size


def split_frame_into_mblocks(self, input_frame):
    global BLOCK_SIZE
    blocks = []
    block_coords = []

    # iterate over blocks
    block_idx = 0
    for i in range(0, vidshape[0], BLOCK_SIZE):
        if (i+BLOCK_SIZE > vidshape[0]):
            break
        for j in range(0, vidshape[1], BLOCK_SIZE):
            if (j+BLOCK_SIZE > vidshape[1]):
                break

            block = input_frame[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

            # we keep all blocks
            blocks.append(block)

            # we keep track of coordinates for later
            block_idx += 1
            block_coords.append([j, i])

    print("Finished splitting frame into macro blocks")
    return [blocks, block_coords]


def find_match(img, block, block_coord):
    global BLOCK_SIZE
    # Returning these
    best_coord = [0, 0]
    best_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

    # OPTIMIZE SEARCH
    # return input location if the same location on reference frame is very similar since motion prediction is
    # based on the assumption that there are many static pixels across frames
    # evaluate input block coord on ref frame
    block_at_input_location = img[block_coord[1]: block_coord[1] +
                                  BLOCK_SIZE, block_coord[0]:block_coord[0]+BLOCK_SIZE]
    diff = np.sum(np.abs(cv2.subtract(block_at_input_location, block)))

    if (diff <= SIMILARITY_THRESHOLD):
        # We got static!
        best_coord = block_coord
        best_block = block_at_input_location
    else:
        # The block isn't static, we start searching
        image = img
        best_match = 9999999999

        # Loop over all blocks in reference frame within the search window
        dist_from_block = SEARCH_WINDOW_SIZE
        # Mins
        i_min = max(block_coord[1] - dist_from_block, 0)
        j_min = max(block_coord[0] - dist_from_block, 0)
        # Max
        i_max = min(block_coord[1] + dist_from_block, vidshape[0])
        j_max = min(block_coord[0] + dist_from_block, vidshape[1])

        # Search for the block
        STEP_SIZE = round(BLOCK_SIZE/3)
        for i in range(i_min, i_max, STEP_SIZE):

            # Check if out of bound of the image
            if (i+BLOCK_SIZE >= i_max):
                continue
            for j in range(j_min, j_max, STEP_SIZE):
                if (j+BLOCK_SIZE >= j_max):
                    continue

                # Valid block within image bound
                ref_block = image[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]

                # Get sum difference between the 2 blocks
                diff = np.sum(np.abs(ref_block - block))

                # Update if it is a better block
                if (diff < best_match):
                    best_match = diff
                    best_coord = [j, i]
                    best_block = ref_block

    return [best_coord, best_block]


def get_motion_vector(match_coord, search_coord):
    # We want vector from search coordinate to match coordinate
    dx = match_coord[0] - search_coord[0]
    dy = match_coord[1] - search_coord[1]
    motion_vector = [dx, dy]
    return motion_vector


def process_motion_prediction(input_frame, ref_frame):
    # Break input frame into 16x16 blocks
    [blocks, block_coords] = split_frame_into_mblocks(input_frame)

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

    return [motion_vectors, block_coords]


def get_residuals(input_frame, reconstructed):
    delta = input_frame - reconstructed
    return delta


def reconstruct_from_motion_vectors(motion_vectors, ref_frame, block_coords, vidshape):
    # We reconstruct an image solely from the motion vectors and the reference frame
    # which we want to compare with the actual input frame to get the residual frame
    reconstruct_img = np.zeros(
        (vidshape[0], vidshape[1], CHANNELS), ref_frame.dtype)
    num_static = 0
    for block_i in range(len(block_coords)):
        [j, i] = block_coords[block_i]
        v = motion_vectors[block_i]
        sampled_block = np.zeros(
            (BLOCK_SIZE, BLOCK_SIZE, CHANNELS), ref_frame.dtype)
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

    # print("There are", num_static, "static blocks out of",
    #       len(block_coords), "blocks")
    return reconstruct_img
