import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

# MODES
TEST_IMAGE = False
TEST_VIDEO = not TEST_IMAGE

######################################################################
# MOTION PREDICTION AND COMPENTSATION

# TESTING
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
img1 = cv2.imread('../images/oscar-cat/5.jpg')
img2 = cv2.imread('../images/oscar-cat/6.jpg')
overlayImg = cv2.imread('../images/oscar-cat/5+6.jpg')

if TEST_CORGI:
    img1 = cv2.imread("../images/corgi-underwater/15.jpg")
    img2 = cv2.imread("../images/corgi-underwater/16.jpg")
    overlayImg = cv2.imread("../images/corgi-underwater/16.jpg")
elif TEST_SIMPLE:
    img1 = cv2.imread("../images/sequences/minor-jump/0.png")
    img2 = cv2.imread("../images/sequences/minor-jump/1.png")
    overlayImg = cv2.imread("../images/sequences/minor-jump/overlay.png")

refImage = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
inputImage = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
overlayImg = cv2.cvtColor(overlayImg, cv2.COLOR_BGR2RGB)
vidshape = [480,  854]
channels = 3

# We want smaller blocks for smaller images to avoid blocky reconstruction
BLOCK_SIZE = 4  # 4 if vidshape[0] * vidshape[1] <= 150000 else 16
print("Block size: ", BLOCK_SIZE)
SEARCH_WINDOW_SIZE = 2 * BLOCK_SIZE

# fig, ((ax_ref, ax_input, ax_overlay), (ax_replaced, ax_residual, ax_reconstructed)) = plt.subplots(
#     2, 3, figsize=(14, 9))
# fig.suptitle("Motion Prediction & Compensation", fontsize=16)

# ax_overlay.imshow(overlayImg)
# ax_overlay.set_title("Overlay Frames with Highlighted Nonstatic Blocks")


# MOTION ESTIMATION
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


def split_frame_into_mblocks(input_frame):
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
    print("Residual input dims:", input_frame.shape, reconstructed.shape)
    delta = input_frame - reconstructed
    return delta


######################################################################
# H.264 FRAMES ENCODING
ENCODING_PATTERN_LENGTH = 4
# Display order of frames: I, B, P, B, P, B, P
# Testing with: I, P, P, P
frames = []
reference_frames = []


class Frame:
    def __init__(self, frame_type, motion_vectors, residuals, block_coords, index, ref_idx):
        self.t = frame_type  # I, P, B
        self.mv = motion_vectors
        self.r = residuals
        self.c = block_coords
        self.i = index
        self.ref_i = ref_idx


def process_I_frame(input_frame, frame_num):
    reference_frames.append(input_frame)
    return Frame("I", None, None, None, frame_num, frame_num % ENCODING_PATTERN_LENGTH)


def process_B_frame(input, frame_num):\
        # TODO: bi-directional
    # ref_idx = frame_num % 2 - 1
    # ref = reference_frames[ref_idx]
    # # Pipeline
    # # 1. Get motion vectors for each block
    # [motion_vecs, coords] = process_motion_prediction(input, ref)
    # # 2. Reconstruct image from reference img and motion vectors
    # reconstructed_img = reconstruct_from_motion_vectors(
    #     motion_vecs, ref, coords)
    # # 3. Get residuals for better result
    # residuals = get_residuals(input_frame=inputImage,
    #                           reconstructed=reconstructed_img)
    # new_frame = Frame("B", motion_vectors=motion_vecs,
    #                   residuals=residuals, block_coords=coords)
    return


def process_P_frame(input, frame_num):
    # TODO: Debug ref idx
    ref_idx = math.floor(frame_num / ENCODING_PATTERN_LENGTH)
    ref = reference_frames[ref_idx]
    print("frame num", frame_num)
    # Pipeline
    # 1. Get motion vectors for each block
    [motion_vecs, coords] = process_motion_prediction(input, ref)
    # 2. Reconstruct image from reference img and motion vectors
    reconstructed_img = reconstruct_from_motion_vectors(
        motion_vecs, ref, coords)
    # 3. Get residuals for better result
    residuals = get_residuals(input_frame=input,
                              reconstructed=reconstructed_img)
    new_frame = Frame("P", motion_vectors=motion_vecs,
                      residuals=residuals, block_coords=coords, index=frame_num, ref_idx=ref_idx)
    return new_frame


def encode_frame(input_frame, frame_num):
    frame_type = ""
    encoded_frame = ""
    if (frame_num % ENCODING_PATTERN_LENGTH == 0):
        encoded_frame = process_I_frame(input_frame, frame_num)
        frame_type = "I"
    else:
        encoded_frame = process_P_frame(input_frame, frame_num)
        frame_type = "P"
    # else:
    #     encoded_frame = process_P_frame(input_frame, frame_num)
    #     frame_type = "P"
    print("Encoded frame of type", frame_type)
    frames.append(encoded_frame)
    return

######################################################################
# DECODING


def reconstruct_from_motion_vectors(motion_vectors, ref_frame, block_coords):
    # We reconstruct an image solely from the motion vectors and the reference frame
    # which we want to compare with the actual input frame to get the residual frame
    reconstruct_img = np.zeros(
        (vidshape[0], vidshape[1], channels), ref_frame.dtype)
    num_static = 0
    for block_i in range(len(block_coords)):
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
        # print("Finished reconstructing block", block_i)

    # print("There are", num_static, "static blocks out of",
    #       len(block_coords), "blocks")
    return reconstruct_img


def fully_reconstruct(residuals, img):
    result = img + residuals
    return result


def reconstruct_P_frame(cur_frame):
    ref = reference_frames[cur_frame.ref_i]
    reconstruct_img = reconstruct_from_motion_vectors(
        cur_frame.mv, ref, cur_frame.c)
    # full_img = fully_reconstruct(residuals=cur_frame.r, img=reconstruct_img)
    return reconstruct_img


def reconstruct_video():
    # Set up video writer
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output.mp4', fourcc, fps,
                          (vidshape[1],  vidshape[0]))
    print("Set up video writer")

    # Iteration variables
    num_frames = len(frames)
    cur_frame_idx = 0
    num_ref_seen = 0
    while (cur_frame_idx < num_frames):
        cur_frame = frames[cur_frame_idx]
        if (cur_frame.t == "I"):
            # we have reference frame, we simply write it
            frame = reference_frames[num_ref_seen]
            # out.write(frame)
            num_ref_seen += 1
        elif (cur_frame.t == "P"):
            # print("Decoding prediction frame type P")
            # we have reference frame, we simply write it
            frame = reconstruct_P_frame(cur_frame)
            out.write(frame)
        cur_frame_idx += 1
    print("Finished writing frames of length", cur_frame_idx + 1)
    out.release()
    return


######################################################################
if TEST_IMAGE:
    # We test by encoding an I frame and a B frame
    encode_frame(refImage, 0)
    encode_frame(inputImage, 1)

######################################################################
if TEST_VIDEO:
    cap = cv2.VideoCapture('../videos/corgi_short.mp4')

    # Define the codec and create VideoWriter object
    # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    # fps = 20.0
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # change dimension for other videos
    # out = cv2.VideoWriter('output.mp4',fourcc, fps, (854,480))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv2.flip(frame, 0)
        encode_frame(frame, frame_num)
        frame_num += 1
        # if (len(residual_frames) > 0):
        #     cv2.imshow('frame', residual_frames[len(residual_frames)-1])
        if cv2.waitKey(1) == ord('q'):
            break

    print("Finished encoding all frames, will decocode and output video")
    reconstruct_video()

    # Release everything if job is finished
    print("Releasing everything. Job finished. ")
    cap.release()

    cv2.destroyAllWindows()


# plt.show()
print("Finished!")
