import cv2
import numpy as np

class FrameNumberDrawer:
    """
    Draws frame numbers safely on each video frame.
    Never crashes even if frames are missing, corrupted, or None.
    """

    def __init__(self, color=(0, 255, 0), scale=1.0, thickness=2):
        self.color = color
        self.scale = scale
        self.thickness = thickness

    def draw(self, frames):
        """
        Draw frame number on top-left corner of each frame.

        Args:
            frames (list): list of numpy frames

        Returns:
            list: list of frames with numbers drawn
        """
        output_frames = []

        for idx, frame in enumerate(frames):

            # SAFETY CHECK: skip None frames
            if frame is None:
                print(f"[FrameNumberDrawer] Warning: Frame {idx} is None. Skipping.")
                output_frames.append(np.zeros((720,1280,3), dtype=np.uint8))  # fallback black frame
                continue

            # SLICE PROTECTION: ensure this is a valid numpy image
            if not isinstance(frame, np.ndarray):
                print(f"[FrameNumberDrawer] Warning: Frame {idx} is not a numpy array. Skipping.")
                output_frames.append(frame)
                continue

            try:
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    str(idx),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.scale,
                    self.color,
                    self.thickness
                )
                output_frames.append(annotated)

            except Exception as e:
                print(f"[FrameNumberDrawer] ERROR drawing frame {idx}: {e}")
                output_frames.append(frame)

        return output_frames