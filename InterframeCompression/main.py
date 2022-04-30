import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

TEST_VIDEO = False
TEST_IMG = not TEST_VIDEO
SAVE_FRAMES = False

fig, ax = plt.subplots()

frames = []
macro_blocks = []

######################################################################


class Frame:
    def __init__(self, frame_type, macro_blocks_idx):
        self.frame_type = frame_type
        self.macro_blocks_idx = macro_blocks_idx


class Macroblock:
    def __init__(self, mb_type, motion_vector):
        self.mb_type = mb_type  # I, P, B
        self.motion_vector = motion_vector  # for P,B blocks we have motion vector


######################################################################

block_size = 8
# Split each frame into macroblocks


def split_frame_into_mblocks(input_frame):
    blocks = []
    # iterate over blocks
    for start_i in range(0, vwidth, block_size):
        for start_j in range(0, vheight, block_size):
            cur_block = np.zeros((block_size, block_size))
            # (start_i, start_j) is the left upper corner of each block
            for i in range(0, block_size):
                if (start_i + i >= vwidth):
                    continue
                for j in range(0, block_size):
                    if (start_j + j >= vheight):
                        continue
                    # we collect each entry in the block
                    cur_block[i][j] = input_frame[start_i + i][start_j + j][0]
            blocks.append(cur_block)
            ax.add_patch(Rectangle((start_i, start_j), block_size,
                         block_size,  edgecolor='black', fill=False, lw=0.5))
    print("Finished splitting frame into macro blocks")
    return blocks

# For each block in each (P, B) frame, we try to find a similar block in an I frame
# Then we can save space by encoding that Macroblock with just a motion vector


def match_block():
    img = imread('https://i.stack.imgur.com/JL2LW.png', pilmode='L')
    temp = imread('https://i.stack.imgur.com/UIUzJ.png', pilmode='L')

    corr = sp.correlate2d(img - img.mean(),
                          temp - temp.mean(),
                          boundary='symm',
                          mode='full')


def process_B_frame():
    return


def process_P_frame():
    return


def process_I_frame():
    return

# Display order of frames: I, B, P, B, P, B, P


order_length = 7


def encode_frame(input_frame, frame_num):
    blocks = split_frame_into_mblocks(input_frame)
    macro_blocks.append(blocks)
    frame_num %= order_length
    frame_type = ""
    if (frame_num == 0):
        frame_type = "I"
    elif frame_num % 2 == 1:
        frame_type = "B"
    else:
        frame_type = "P"
    print("Encode frame of type", frame_type)
    new_frame = Frame(frame_type, len(macro_blocks)-1)
    frames.append(new_frame)
    return


######################################################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if TEST_IMG:
    img1 = cv2.imread('../images/corgi-underwater/12.jpg')
    img2 = cv2.imread('../images/corgi-underwater/20.jpg')
    vwidth, vheight, channels = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    encode_frame(img1, 0)
    encode_frame(img2, 1)

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
    # plt.show()


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
