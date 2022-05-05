import numpy as np
import cv2

# MOTION ESTIMATION
# Functions related to processing motion used by both encoder and decoder

# Optimization Params
SIMILARITY_THRESHOLD = 2000
CHANNELS = 3

WRITE_STATIC_BLOCK = True


class MotionProcessor:
    def __init__(self, block_size, shape):
        self.block_size = block_size
        self.shape = shape
        self.search_window_size = block_size * 2

    def process_motion_prediction(self, input_frame, ref_frame):
        # Break input frame into blocks
        [blocks, block_coords] = self._split_frame_into_mblocks(input_frame)

        # For each block, we get its motion vector to ref frame (if it is not static)
        motion_vectors = []
        for block_i in range(len(blocks)):
            searchCoord = block_coords[block_i]

            # We find the matching block in reference frame
            [best_coord, best_block] = self._find_match(
                ref_frame, blocks[block_i], searchCoord)
            motion_vector = self._get_motion_vector(best_coord, searchCoord)

            motion_vectors.append(motion_vector)

        return [motion_vectors, block_coords]

    def get_residuals(self, input_frame, reconstructed):
        delta = input_frame - reconstructed
        return delta

    def reconstruct_from_motion_vectors(self, motion_vectors, ref_frame, block_coords):
        # We reconstruct an image solely from the motion vectors and the reference frame
        # which we want to compare with the actual input frame to get the residual frame
        reconstruct_img = np.zeros(
            (self.shape[0], self.shape[1], CHANNELS), ref_frame.dtype)
        num_static = 0
        for block_i in range(len(block_coords)):
            [j, i] = block_coords[block_i]
            v = motion_vectors[block_i]
            sampled_block = np.zeros(
                (self.block_size, self.block_size, CHANNELS), ref_frame.dtype)
            if (v[0] == 0 and v[1] == 0 and WRITE_STATIC_BLOCK):
                # We find static blocks and directly sample from the reference frame
                sampled_block = ref_frame[i:i +
                                          self.block_size, j:j+self.block_size]
                num_static += 1
            else:
                # For nonstatic blocks, we find the motion predicted block
                i_0 = i + v[1]
                j_0 = j + v[0]
                sampled_block = ref_frame[i_0:i_0 +
                                          self.block_size, j_0:j_0+self.block_size]
            reconstruct_img[i:i+self.block_size, j:j +
                            self.block_size] = sampled_block

        print("There are", num_static, "static blocks out of",
              len(block_coords), "blocks")
        return reconstruct_img

    #################################################################
    # PRIVATE METHODS

    def _split_frame_into_mblocks(self, input_frame):
        # print("Splitting frame into mblocks with input of shape", input_frame.shape)
        # print("Self shape", self.shape)
        blocks = []
        block_coords = []

        # iterate over blocks
        block_idx = 0
        for i in range(0, self.shape[0], self.block_size):
            if (i+self.block_size > self.shape[0]):
                break
            for j in range(0, self.shape[1], self.block_size):
                if (j+self.block_size > self.shape[1]):
                    break

                block = input_frame[i:i+self.block_size, j:j+self.block_size]

                # we keep all blocks
                blocks.append(block)
                # we keep track of coordinates for later
                block_idx += 1
                block_coords.append([j, i])

        # print("Finished splitting frame into macro blocks")
        return [blocks, block_coords]

    def _find_match(self, img, block, block_coord):
        # Returning these
        best_coord = [0, 0]
        best_block = np.zeros((self.block_size, self.block_size))

        # OPTIMIZE SEARCH
        # return input location if the same location on reference frame is very similar since motion prediction is
        # based on the assumption that there are many static pixels across frames
        # evaluate input block coord on ref frame
        block_at_input_location = img[block_coord[1]: block_coord[1] +
                                      self.block_size, block_coord[0]:block_coord[0]+self.block_size]
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
            dist_from_block = self.search_window_size
            # Mins
            i_min = max(block_coord[1] - dist_from_block, 0)
            j_min = max(block_coord[0] - dist_from_block, 0)
            # Max
            i_max = min(block_coord[1] + dist_from_block, self.shape[0])
            j_max = min(block_coord[0] + dist_from_block, self.shape[1])

            # Search for the block
            STEP_SIZE = round(self.block_size/3)
            for i in range(i_min, i_max, STEP_SIZE):

                # Check if out of bound of the image
                if (i+self.block_size >= i_max):
                    continue
                for j in range(j_min, j_max, STEP_SIZE):
                    if (j+self.block_size >= j_max):
                        continue

                    # Valid block within image bound
                    ref_block = image[i:i+self.block_size, j:j+self.block_size]

                    # Get sum difference between the 2 blocks
                    diff = np.sum(np.abs(ref_block - block))

                    # Update if it is a better block
                    if (diff < best_match):
                        best_match = diff
                        best_coord = [j, i]
                        best_block = ref_block

        return [best_coord, best_block]

    def _get_motion_vector(self, match_coord, search_coord):
        # We want vector from search coordinate to match coordinate
        dx = match_coord[0] - search_coord[0]
        dy = match_coord[1] - search_coord[1]
        motion_vector = [dx, dy]
        return motion_vector
