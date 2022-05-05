import cv2
from motion import MotionProcessor
from DCTcompressor import DCTCompressor

######################################################################
# DECODING

WRITE_REF_FRAMES = True


class Decoder:
    def __init__(self, encoded_frames, fps, shape, ref_frames, block_size, with_DCT):
        self.encoded_frames = encoded_frames
        self.fps = fps
        self.shape = shape
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.ref_frames = ref_frames
        self.MotionProcessor = MotionProcessor(
            block_size=block_size, shape=shape)
        self.DCTCompressor = DCTCompressor(block_size=block_size)
        self.with_DCT = with_DCT

    def reconstruct_video(self, with_residuals):
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
            if (cur_frame.t == "I" and WRITE_REF_FRAMES):
                # we have reference frame, we simply write it
                frame = self.ref_frames[num_ref_seen]
                out.write(frame)
                num_ref_seen += 1
            elif (cur_frame.t == "P"):
                # we have P frame, we reconstruct P frame
                frame = self._reconstruct_P_frame(cur_frame, with_residuals)
                out.write(frame)
            cur_frame_idx += 1
        print("Finished writing frames of length", cur_frame_idx + 1)
        out.release()
        return

    #################################################################
    # PRIVATE METHODS

    def _fully_reconstruct(self, residuals, img):
        # We need to transform residuals back
        if (self.with_DCT):
            decompressed_residuals = self.DCTCompressor.decompress(
                compressed=residuals, imshape=img.shape)
            result = img + decompressed_residuals
            return result
        else:
            return img + residuals

    def _reconstruct_P_frame(self, cur_frame, with_residuals):
        ref = self.ref_frames[cur_frame.ref_i]
        reconstruct_img = self.MotionProcessor.reconstruct_from_motion_vectors(
            cur_frame.mv, ref, cur_frame.c)
        if (with_residuals):
            return self._fully_reconstruct(residuals=cur_frame.r, img=reconstruct_img)
        else:
            return reconstruct_img
