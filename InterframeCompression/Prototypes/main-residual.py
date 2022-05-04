import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import scipy.signal as sp

TEST_VIDEO = False
TEST_IMG = not TEST_VIDEO
SAVE_FRAMES = False
OUTPUT_COMPRESSION = False

residual_frames = []
reference_frames = [] # "Coded Picture Buffer" in the book 
# We keep the last full frame at all times so we can get the delta easily
last_full_frame = cv2.imread('../images/corgi-underwater/11.jpg',
                             cv2.IMREAD_GRAYSCALE)  # Used when encoding

fig, ((ax_prev_frame, ax_cur_frame, ax_residual)) = plt.subplots(1, 3,
                                                                 figsize=(10, 5))

######################################################################

# We have 1 reference frame followed by 6 residual frames
REFERENCE_SET_LEN = 7

# Returns the residual frame from input frame


def process_residual_frame(input_frame):
    global last_full_frame

    delta = input_frame - last_full_frame
    ax_residual.imshow(delta, cmap='gray')
    ax_residual.set_title('Current Residual')
    ax_residual.set_axis_off()

    ax_prev_frame.imshow(last_full_frame, cmap='gray')
    ax_prev_frame.set_title('Prev Full Frame')
    ax_prev_frame.set_axis_off()

    ax_cur_frame.imshow(input_frame, cmap='gray')
    ax_cur_frame.set_title('Current Full Frame')
    ax_cur_frame.set_axis_off()

    return delta


def encode_frame(input_frame, frame_num):
    global last_full_frame
    if (frame_num % REFERENCE_SET_LEN == 0):
        reference_frames.append(input_frame)
    else:
        residual_frames.append(process_residual_frame(input_frame))
    last_full_frame = input_frame
    print("encoded frame index", frame_num)
    return


######################################################################

# We decode with the reference frames array and residuals array
def decode_residuals():
    # Set up video writer
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (854, 480))
    print("Set up video writer")

    # Iteration variables
    num_frames = len(reference_frames) + len(residual_frames)
    cur_frame_idx = 0
    last_full_frame = reference_frames[0]
    num_ref_seen = 0
    num_res_seen = 0
    while (cur_frame_idx < num_frames):
        if (cur_frame_idx % REFERENCE_SET_LEN == 0):
            print("Decoding reference frame")
            # we have reference frame, we simply write it
            frame = reference_frames[num_ref_seen]
            out.write(frame)
            last_full_frame = frame
            num_ref_seen += 1
        else:
            # we grab residual and last full frame
            print("Decoding residual frame")
            res = residual_frames[num_res_seen]
            frame = last_full_frame + res
            last_full_frame = frame
            out.write(frame)
            num_res_seen += 1
        cur_frame_idx += 1
    print("Finished writing frames of length", cur_frame_idx + 1)
    out.release()


######################################################################


if TEST_IMG:
    img1 = cv2.imread('../images/corgi-underwater/11.jpg',
                      cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../images/corgi-underwater/18.jpg',
                      cv2.IMREAD_GRAYSCALE)
    vheight, vwidth = img1.shape
    encode_frame(img1, 0)
    encode_frame(img2, 1)

    if OUTPUT_COMPRESSION:
        # Save compressed data of frames
        json_string = json.dumps([ob.__dict__ for ob in residual_frames])
        with open("frames.json", "w") as text_file:
            print(f"{json_string}", file=text_file)
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
        # frame = cv2.flip(frame, 0)
        encode_frame(frame, frame_num)
        frame_num += 1
        # out.write(frame)
        if (len(residual_frames) > 0):
            cv2.imshow('frame', residual_frames[len(residual_frames)-1])
        if cv2.waitKey(1) == ord('q'):
            break

    print("Finished encoding all frames, will decocode and output video")
    decode_residuals()

    # Release everything if job is finished
    print("Releasing everything. Job finished. ")
    cap.release()

    cv2.destroyAllWindows()
