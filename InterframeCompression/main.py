import cv2
from encoder import Encoder
from decoder import Decoder

######################################################################
# TESTS
CAT_PATH = "../videos/cat_crop.mp4"
FEW_PATH = "../videos/cat_extreme_cut.mp4"
TRAFFIC_PATH = "../videos/traffic_cut.mp4"
CAMERA_PAN_PATH = "../videos/camera_pan.mp4"
######################################################################
# PARAMS
VIDEO_INPUT = CAMERA_PAN_PATH
FRAME_RATE = 25
BLOCK_SIZE = 8
ENCODING_PATTERN = ["I", "P", "P", "P"]  # ON BOOK: I, B, P, B, P, B, P

######################################################################
# GET INPUT DATA
cap = cv2.VideoCapture(VIDEO_INPUT)
width = int(cap.get(3))
height = int(cap.get(4))
vidshape = [height, width]

######################################################################
# ENCODE VIDEO
WITH_RESIDUAL = True
WITH_DCT = WITH_RESIDUAL and True
WITH_QUANTIZE = WITH_DCT and True
h264_encoder = Encoder(pattern=ENCODING_PATTERN,
                       shape=vidshape, block_size=BLOCK_SIZE, with_DCT=WITH_DCT, with_Q = WITH_QUANTIZE)

# READ AND ENCODE EVERY FRAME
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Encoder saves all encoded frames
    h264_encoder.encode_frame(frame, frame_num)
    frame_num += 1
    if cv2.waitKey(1) == ord('q'):
        break
print("Finished encoding all frames, will decocode and output video.")

######################################################################
# DECODE VIDEO
h264_decoder = Decoder(encoded_frames=h264_encoder.encoded_frames, fps=25.0, shape=vidshape,
                       ref_frames=h264_encoder.ref_frames, block_size=BLOCK_SIZE, with_DCT=WITH_DCT, with_Q=WITH_QUANTIZE)
h264_decoder.reconstruct_video(with_residuals=WITH_RESIDUAL)

######################################################################
# DONE
print("Releasing everything. Job finished. ")
cap.release()
cv2.destroyAllWindows()
h264_decoder.MotionProcessor.print_performance()
print("Finished!")
