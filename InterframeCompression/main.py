import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import scipy.signal as sp
import math

TEST_VIDEO = False
TEST_IMG = not TEST_VIDEO
SAVE_FRAMES = False
OUTPUT_COMPRESSION = False

frames = []
macro_blocks = []
reference_frames = []

fig, ((ax_orig, ax_template_frame), (ax_template, ax_corr)) = plt.subplots(2, 2,
                                                                           figsize=(6, 15))

######################################################################


class Frame:
    def __init__(self, frame_type, macro_blocks_idx):
        self.t = frame_type  # I, P, B
        self.i = macro_blocks_idx  # index into the macro_blocks array


class Macroblock:
    def __init__(self, mb_type, motion_vector):
        self.mb_type = mb_type  # I, P, B
        self.motion_vector = motion_vector  # for P,B blocks we have motion vector


######################################################################

block_size = 32
# Split each frame into macroblocks


def split_frame_into_mblocks(input_frame):
    num_blocks_hor = math.ceil(vwidth / block_size)
    num_blocks_vert = math.ceil(vheight / block_size)
    blocks = np.zeros(
        (num_blocks_hor, num_blocks_vert, block_size, block_size))
    print("Block size", blocks.shape)
    print("Input size (y, x)", input_frame.shape)
    print("Vwidth height", vwidth, vheight)
    # iterate over blocks
    for col in range(0, num_blocks_hor):
        start_col = col * block_size
        for row in range(0, num_blocks_vert):
            start_row = row * block_size
            # (start_i, start_j) is the left upper corner of each block
            for i in range(0, block_size):
                if (start_col + i >= vwidth):
                    continue
                for j in range(0, block_size):
                    if (start_row + j >= vheight):
                        continue
                    # we collect each entry in the block
                    # print("input x, y", start_col + i, start_row + j)
                    blocks[col][row][j][i] = input_frame[start_row +
                                                         j][start_col + i]
            # draw out bounding box for each block
            ax_template_frame.add_patch(Rectangle((start_col, start_row), block_size,
                                                  block_size,  edgecolor='black', fill=False, lw=0.5))
    print("Finished splitting frame into macro blocks")
    return blocks

# For each block in each (P, B) frame, we try to find a similar block in an I frame
# Then we can save space by encoding that Macroblock with just a motion vector


def match_block(img, temp):
    print(img.shape, "temp shape", temp.shape)
    corr = sp.correlate2d(img - img.mean(),
                          temp - temp.mean(),
                          boundary='symm',
                          mode='full')
    max_coords = np.where(corr == np.max(corr))

    ax_corr.imshow(corr, cmap='gray')
    ax_corr.set_title('Cross-correlation')
    ax_corr.set_axis_off()
    ax_corr.plot(max_coords[1], max_coords[0], 'c*', markersize=5)

    return [max_coords[1], max_coords[0]]


def process_B_frame(input_frame, frame_num):
    # Find closest reference frame
    ref_frame = reference_frames[frames[frame_num -
                                        frame_num % order_length].i]
    cur_blocks = split_frame_into_mblocks(input_frame)

    block_i = 1
    block_j = 0
    x, y = match_block(ref_frame, cur_blocks[block_i][block_j])
    block_coord = [block_i * block_size, block_j * block_size]
    # motion_vector = [x - block_coord[0], y - block_coord[1]]
    # mb = Macroblock("B", )

    # plot reference frame
    ax_orig.imshow(ref_frame, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()

    # plot input frame
    ax_template_frame.imshow(input_frame, cmap="gray")
    ax_template_frame.set_title("Full Template Frame")
    ax_template_frame.set_axis_off()
    ax_template_frame.add_patch(Rectangle((block_coord[0], block_coord[1]), block_size,
                                block_size,  edgecolor='red', fill=False, lw=2))

    # plot template (macroblock we are searching for)
    ax_template.imshow(cur_blocks[block_i][block_j], cmap='gray')
    ax_template.set_title('Template Block')
    ax_template.set_axis_off()

    # draw the match
    ax_orig.plot(x, y, 'ro')
    print("Finished processing B frame")

    return


def process_P_frame(input_frame, frame_num):
    return


def process_I_frame(input_frame):
    reference_frames.append(input_frame)
    return Frame("I", len(reference_frames) - 1)

# Display order of frames: I, B, P, B, P, B, P


order_length = 7


def encode_frame(input_frame, frame_num):
    frame_num %= order_length
    frame_type = ""
    encoded_frame = ""
    if (frame_num == 0):
        encoded_frame = process_I_frame(input_frame)
        frame_type = "I"
    elif frame_num % 2 == 1:
        encoded_frame = process_B_frame(input_frame, frame_num)
        frame_type = "B"
    else:
        encoded_frame = process_P_frame(input_frame, frame_num)
        frame_type = "P"
    print("Encoded frame of type", frame_type)
    frames.append(encoded_frame)
    return


######################################################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if TEST_IMG:
    img1 = cv2.imread('../images/corgi-underwater/12.jpg',
                      cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../images/corgi-underwater/16.jpg',
                      cv2.IMREAD_GRAYSCALE)
    vheight, vwidth = img1.shape
    print("Num blocks (hor, vert):", vwidth / block_size, vheight / block_size)
    encode_frame(img1, 0)
    encode_frame(img2, 1)

    if OUTPUT_COMPRESSION:
        # Save compressed data of frames
        json_string = json.dumps([ob.__dict__ for ob in frames])
        with open("frames.json", "w") as text_file:
            print(f"{json_string}", file=text_file)

    if SAVE_FRAMES:
        # Save frames
        # TODO: replace with compressed inter frame data
        # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        json_dump = json.dumps({'a': macro_blocks, 'aa': [2, (2, 3, 4), macro_blocks], 'bb': [2]},
                               cls=NumpyEncoder)
        with open("blocks.json", "w")as text_file:
            print(f"{json_dump}", file=text_file)
    plt.show()


######################################################################
if TEST_VIDEO:
    cap = cv2.VideoCapture('../videos/corgi_short.mp4')

    # Define the codec and create VideoWriter object
    # https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    # fps = 20.0
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # change dimension for other videos
    # out = cv2.VideoWriter('output.mp4',fourcc, fps, (854,480))

    vwidth, vheight = 0, 0
    if cap.isOpened():
        vwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("Dimensions", vwidth, vheight)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # print("Flipping frame")
        # frame = cv2.flip(frame, 0)
        # write the flipped frame
        encode_frame(frame, frame_num)
        frame_num += 1
        # out.write(frame)
        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release everything if job is finished
    print("Releasing everything. Job finished. ")
    cap.release()
    # out.release()

    cv2.destroyAllWindows()
