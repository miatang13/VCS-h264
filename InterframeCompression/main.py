from base64 import encode
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import motion

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


# MOTION ESTIMATION


######################################################################
# H.264 FRAMES ENCODING

# Display order of frames: I, B, P, B, P, B, P
# Testing with: I, P, P, P


class Frame:
    def __init__(self, frame_type, motion_vectors, residuals, block_coords, index, ref_idx):
        self.t = frame_type  # I, P, B
        self.mv = motion_vectors
        self.r = residuals
        self.c = block_coords
        self.i = index
        self.ref_i = ref_idx


class Encoder:

    def __init__(self, pattern):
        self.ref_frames = []
        self.pattern = pattern
        self.ENCODING_PATTERN_LENGTH = len(pattern)

    def encode_frame(self, input_frame, frame_num):
        frame_type = ""
        encoded_frame = ""
        if (frame_num % self.ENCODING_PATTERN_LENGTH == 0):
            encoded_frame = self._process_I_frame(input_frame, frame_num)
            frame_type = "I"
        else:
            encoded_frame = self._process_P_frame(input_frame, frame_num)
            frame_type = "P"
        # else:
        #     encoded_frame = process_P_frame(input_frame, frame_num)
        #     frame_type = "P"
        print("Encoded frame of type", frame_type)
        return encoded_frame

    #################################################################
    # PRIVATE METHODS

    def _process_I_frame(self, input_frame, frame_num):
        self.ref_frames.append(input_frame)
        return Frame("I", None, None, None, frame_num, frame_num % self.ENCODING_PATTERN_LENGTH)

    def _process_B_frame(self, input, frame_num):
        # TODO: bi-directional
        return

    def _process_P_frame(self, input, frame_num):
        ref_idx = math.floor(frame_num / self.ENCODING_PATTERN_LENGTH)
        ref = self.ref_frames[ref_idx]
        # Pipeline
        # 1. Get motion vectors for each block
        [motion_vecs, coords] = process_motion_prediction(input, ref)
        # 2. Reconstruct image from reference img and motion vectors
        reconstructed_img = motion.reconstruct_from_motion_vectors(
            motion_vecs, ref, coords)
        # 3. Get residuals for better result
        residuals = get_residuals(input_frame=input,
                                  reconstructed=reconstructed_img)
        new_frame = Frame("P", motion_vectors=motion_vecs,
                          residuals=residuals, block_coords=coords, index=frame_num, ref_idx=ref_idx)
        return new_frame


######################################################################
# DECODING

class Decoder:
    def __init__(self, encoded_frames, fps, shape, ref_frames):
        self.encoded_frames = encoded_frames
        self.fps = fps
        self.shape = shape
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.ref_frames = ref_frames

    def reconstruct_video(self):
        # Set up video writer
        out = cv2.VideoWriter('output.mp4', self.fourcc, self.fps,
                              (self.shape[1], self.shape[0]))
        print("Set up video writer")

        # Iteration variables
        num_frames = len(self.encoded_frames)
        cur_frame_idx = 0
        num_ref_seen = 0
        while (cur_frame_idx < num_frames):
            cur_frame = self.encoded_frames[cur_frame_idx]
            if (cur_frame.t == "I"):
                # we have reference frame, we simply write it
                frame = self.ref_frames[num_ref_seen]
                out.write(frame)
                num_ref_seen += 1
            elif (cur_frame.t == "P"):
                # we have P frame, we reconstruct P frame
                frame = self._reconstruct_P_frame(cur_frame)
                out.write(frame)
            cur_frame_idx += 1
        print("Finished writing frames of length", cur_frame_idx + 1)
        out.release()
        return

    #################################################################
    # PRIVATE METHODS

    def fully_reconstruct(residuals, img):
        result = img + residuals
        return result

    def _reconstruct_P_frame(self, cur_frame):
        ref = self.ref_frames[cur_frame.ref_i]
        reconstruct_img = motion.reconstruct_from_motion_vectors(
            cur_frame.mv, ref, cur_frame.c)
        # full_img = fully_reconstruct(residuals=cur_frame.r, img=reconstruct_img)
        return reconstruct_img


######################################################################
if TEST_IMAGE:
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

    # We test by encoding an I frame and a P frame
    encode_frame(refImage, 0)
    encode_frame(inputImage, 1)

######################################################################
if TEST_VIDEO:
    CORGI_DIM = [480,  854]
    CAT_DIM = [600, 600]
    TRAFFIC_DIM = [360, 640]
    CORGI_PATH = '../videos/corgi_short.mp4'
    CAT_PATH = "../videos/cat_crop.mp4"
    TRAFFIC_PATH = "../videos/traffic_cut.mp4"
    VIDEO_INPUT = TRAFFIC_PATH
    vidshape = TRAFFIC_DIM
    FRAME_RATE = 25

    cap = cv2.VideoCapture(VIDEO_INPUT)

    frame_num = 0
    encoded_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = encode_frame(frame, frame_num)
        encoded_frames.append(frame)
        frame_num += 1
        if cv2.waitKey(1) == ord('q'):
            break

    print("Finished encoding all frames, will decocode and output video")
    reconstruct_video(encoded_frames)

    print("Releasing everything. Job finished. ")
    cap.release()

    cv2.destroyAllWindows()


# plt.show()
print("Finished!")
