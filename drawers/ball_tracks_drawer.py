from .utils import draw_traingle

class BallTracksDrawer:
    """
    Draws ball tracks/pointers on video frames.
    Handles missing frames, missing keys, and malformed bounding boxes safely.
    """

    def __init__(self, color=(0, 255, 0)):
        self.ball_pointer_color = color

    def draw(self, video_frames, tracks):
        """
        Draws ball pointers for each frame.

        Args:
            video_frames (list): list of numpy frames
            tracks (list): list of dicts (each dict = detected balls)

        Returns:
            list: frames with ball pointers drawn
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):

            # Clone frame
            frame = frame.copy()

            # Get ball dict safely
            ball_dict = None
            try:
                ball_dict = tracks[frame_num]
            except Exception:
                ball_dict = {}

            # If no detections, just append frame
            if not isinstance(ball_dict, dict) or len(ball_dict) == 0:
                output_video_frames.append(frame)
                continue

            # Iterate through detected balls
            for _, ball in ball_dict.items():

                # Ensure 'bbox' exists
                bbox = ball.get("bbox", None)

                if bbox is None or len(bbox) != 4:
                    # Skip malformed bounding boxes
                    continue

                # Draw pointer for this ball
                try:
                    frame = draw_traingle(frame, bbox, self.ball_pointer_color)
                except Exception as e:
                    print(f"[BallTracksDrawer] Warning: draw_traingle failed: {e}")

            output_video_frames.append(frame)

        return output_video_frames