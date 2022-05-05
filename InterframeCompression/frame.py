class Frame:
    def __init__(self, frame_type, motion_vectors, residuals, block_coords, index, ref_idx):
        self.t = frame_type  # I, P, B
        self.mv = motion_vectors
        self.r = residuals
        self.c = block_coords
        self.i = index
        self.ref_i = ref_idx
