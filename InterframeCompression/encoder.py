from motion import MotionProcessor
from frame import Frame
import math


class Encoder:

    def __init__(self, pattern, shape, block_size):
        self.ref_frames = []
        self.encoded_frames = []
        self.pattern = pattern
        self.ENCODING_PATTERN_LENGTH = len(pattern)
        self.MotionProcessor = MotionProcessor(
            block_size=block_size, shape=shape)

    def encode_frame(self, input_frame, frame_num):
        print("Encoding new frame of index", frame_num)
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
        self.encoded_frames.append(encoded_frame)
        return

    #################################################################
    # PRIVATE METHODS

    def _process_I_frame(self, input_frame, frame_num):
        self.ref_frames.append(input_frame)
        return Frame("I", None, None, None, frame_num, frame_num % self.ENCODING_PATTERN_LENGTH)

    def _process_B_frame(self, input, frame_num):
        # TODO: bi-directional
        return

    def _process_P_frame(self, input, frame_num):
        print("Processing P frame")
        ref_idx = math.floor(frame_num / self.ENCODING_PATTERN_LENGTH)
        ref = self.ref_frames[ref_idx]
        # Pipeline
        # 1. Get motion vectors for each block
        [motion_vecs, coords] = self.MotionProcessor.process_motion_prediction(
            input, ref)
        print("Finished processing motion. Got motion vectors.")
        # 2. Reconstruct image from reference img and motion vectors
        reconstructed_img = self.MotionProcessor.reconstruct_from_motion_vectors(
            motion_vecs, ref, coords)
        # 3. Get residuals for better result
        residuals = self.MotionProcessor.get_residuals(input_frame=input,
                                                       reconstructed=reconstructed_img)
        new_frame = Frame("P", motion_vectors=motion_vecs,
                          residuals=residuals, block_coords=coords, index=frame_num, ref_idx=ref_idx)
        return new_frame
