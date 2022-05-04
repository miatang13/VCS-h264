import cv2
from motion import MotionProcessor

######################################################################
# DECODING


class Decoder:
    def __init__(self, encoded_frames, fps, shape, ref_frames, block_size):
        self.encoded_frames = encoded_frames
        self.fps = fps
        self.shape = shape
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.ref_frames = ref_frames
        self.MotionProcessor = MotionProcessor(
            block_size=block_size, shape=shape)

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

    def _fully_reconstruct(residuals, img):
        result = img + residuals
        return result

    def _reconstruct_P_frame(self, cur_frame):
        ref = self.ref_frames[cur_frame.ref_i]
        reconstruct_img = self.MotionProcessor.reconstruct_from_motion_vectors(
            cur_frame.mv, ref, cur_frame.c)
        # full_img = _fully_reconstruct(residuals=cur_frame.r, img=reconstruct_img)
        return reconstruct_img
