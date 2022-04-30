import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEST_VIDEO = False
TEST_IMG = not TEST_VIDEO

fig, ax = plt.subplots()

frames = []

######################################################################


class Frame:
    def __init__(self, frame_type, macro_blocks):
        self.frame_type = frame_type
        self.macro_blocks = macro_blocks


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

# For each block in each (P, B) frame, we try to find a similar block in an I frame
# Then we can save space by encoding that Macroblock with just a motion vector

# Display order of frames: I, B, P, B, P, B, P


order_length = 7


def encode_frame(input_frame, frame_num):
    macro_blocks = split_frame_into_mblocks(input_frame)
    frame_num %= order_length
    frame_type = ""
    if (frame_num == 0):
        frame_type = "I"
    elif frame_num % 2 == 1:
        frame_type = "B"
    else:
        frame_type = "P"
    print("Encode frame of type", frame_type)
    new_frame = Frame(frame_type, macro_blocks)
    frames.append(new_frame)
    return


######################################################################
if TEST_IMG:
    img = cv2.imread('../images/happy-corgi(8x8).jpg')
    vwidth, vheight, channels = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    encode_frame(img, 0)
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
