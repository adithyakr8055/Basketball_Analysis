import numpy as np
import supervision as sv
import cv2 # <- ADDED: Required for the requested replacement logic

def _normalize_keypoints(keypoints):
    """
    Convert various keypoint formats to a Nx2 numpy array of (x, y) coordinates.
    Accepts:
      - None -> returns None
      - numpy arrays (Nx2 or NxK) -> returns Nx2
      - PyTorch tensors -> converts to numpy
      - objects with `.xy` attribute (like some model outputs) -> tries .xy -> to numpy
      - list of tuples/lists -> converts to numpy
    Returns:
      - numpy array shape (N,2) or None if conversion not possible / empty
    """
    if keypoints is None:
        return None

    # If it's supervision KeyPoints object it might have .xy or .xy.numpy()
    try:
        # prefer .xy attribute if available
        if hasattr(keypoints, "xy"):
            kp = keypoints.xy
            # if kp is tensor-like convert
            if hasattr(kp, "cpu") and hasattr(kp, "numpy"):
                kp = kp.cpu().numpy()
            return np.asarray(kp).reshape(-1, 2) if len(np.asarray(kp)) else None
    except Exception:
        pass

    # If PyTorch tensor
    try:
        if hasattr(keypoints, "cpu") and hasattr(keypoints, "numpy"):
            kp = keypoints.cpu().numpy()
            arr = np.asarray(kp)
            return arr.reshape(-1, 2) if arr.size else None
    except Exception:
        pass

    # If numpy array
    try:
        arr = np.asarray(keypoints)
        if arr.size == 0:
            return None
        # If array is shape (N, >=2) take first two columns
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2].astype(float)
        # If flat list like [x1,y1,x2,y2,...]
        if arr.ndim == 1 and arr.size % 2 == 0:
            return arr.reshape(-1, 2).astype(float)
    except Exception:
        pass

    # If list of (x,y) tuples / lists
    try:
        if isinstance(keypoints, (list, tuple)):
            if len(keypoints) == 0:
                return None
            arr = np.array(keypoints, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    except Exception:
        pass

    # Unknown format
    return None

# NEW HELPER FUNCTION: converts keypoints to list of (int, int) tuples for cv2 drawing
def _kp_to_list(frame_kp):
    """Return list of (x,y) from various formats."""
    if frame_kp is None:
        return []
    if hasattr(frame_kp, "xy"):
        try:
            arr = np.asarray(frame_kp.xy)
            if arr.ndim == 3:
                # Handle batch dimension if present
                arr = arr[0]
            # Convert to list of (int, int) tuples
            return [(int(x), int(y)) for x, y in arr.tolist()]
        except Exception:
            return []
    if isinstance(frame_kp, np.ndarray):
        try:
            arr = frame_kp
            if arr.ndim == 3:
                # Handle batch dimension if present
                arr = arr[0]
            # Reshape and convert to list of (int, int) tuples
            return [(int(x), int(y)) for x, y in arr.reshape(-1, 2).tolist()]
        except Exception:
            return []
    try:
        # list-like
        return [(int(x), int(y)) for x, y in frame_kp]
    except Exception:
        return []


class CourtKeypointDrawer:
    """
    Draws court keypoints on frames, robustly handles missing / malformed keypoints.
    """
    def __init__(self, keypoint_hex='#ff2c2c', radius=8, label_scale=0.5, label_thickness=1):
        self.keypoint_color = keypoint_hex
        self.radius = radius
        self.label_scale = label_scale
        self.label_thickness = label_thickness
        
        # Convert hex to BGR tuple for OpenCV (b, g, r)
        # Assuming supervision Color class is available for safe conversion
        sv_color = sv.Color.from_hex(self.keypoint_color)
        self.bgr_color = (sv_color.b, sv_color.g, sv_color.r)
        
        # NOTE: Removed sv.VertexAnnotator and sv.VertexLabelAnnotator initialization
        # as the drawing is now done directly with cv2.

    def draw(self, frames, court_keypoints):
        """
        Args:
            frames (list): list of numpy images
            court_keypoints (list): list where each element corresponds to the frame's keypoints
        Returns:
            list: annotated frames (same length as input frames)
        """
        output_frames = []

        # guard: if court_keypoints is None or not matching length, allow best-effort
        if court_keypoints is None:
            print("[CourtKeypointDrawer] DBG: court_keypoints is None -> returning frames unchanged")
            return [f.copy() for f in frames]

        # Initialize placeholder for frame keypoints if the input list is too short
        num_frames = len(frames)
        if len(court_keypoints) < num_frames:
             court_keypoints = court_keypoints + [None] * (num_frames - len(court_keypoints))
        elif len(court_keypoints) > num_frames:
             court_keypoints = court_keypoints[:num_frames]


        # Define the offset and color as requested in the replacement
        # Note: The requested replacement logic is simplified and does not
        # include labeling or color conversion, so we will use a basic red circle.
        # offset_x and offset_y are assumed to be 0 unless needed for tactical view overlay
        # Since this is a standalone drawer, we set offset to 0.
        offset_x, offset_y = 0, 0
        
        # Use a fixed color for the drawing as requested: (0, 0, 255) is red in BGR
        draw_color = (0, 0, 255) 
        draw_radius = 3 # Use 3 as requested in the cv2 example
        draw_thickness = -1 # Solid circle

        for idx, frame in enumerate(frames):
            annotated = frame.copy()

            # safe get keypoints for this frame
            frame_keypoints_for_frame = None
            try:
                frame_keypoints_for_frame = court_keypoints[idx]
            except IndexError:
                # Should be fixed by the length check above, but as a safeguard
                print(f"[CourtKeypointDrawer] DBG: no keypoints for frame index {idx}; skipping annotation")
                output_frames.append(annotated)
                continue

            # NEW LOGIC START: Use the custom conversion function
            kp_list = _kp_to_list(frame_keypoints_for_frame)
            
            if not kp_list:
                # nothing to draw
                output_frames.append(annotated)
                continue

            # Draw the points using cv2.circle as requested
            try:
                # NOTE: The requested replacement logic for drawing a circle is below:
                # for idx, (x, y) in enumerate(kp_list):
                #     cv2.circle(out_frame, (x + offset_x, y + offset_y), 3, (0, 0, 255), -1)
                
                # We use 'annotated' as 'out_frame'
                for _, (x, y) in enumerate(kp_list):
                    # Draw a solid circle at each keypoint
                    cv2.circle(
                        annotated, 
                        (x + offset_x, y + offset_y), 
                        draw_radius, 
                        draw_color, 
                        draw_thickness
                    )
                
                # NOTE: The requested replacement does NOT include labeling, 
                # so the label annotator is intentionally omitted here.
                
            except Exception as e:
                print(f"[CourtKeypointDrawer] Warning: cv2 drawing failed on frame {idx}: {e}")
                # continue with previous frame copy
            
            # NEW LOGIC END
            
            output_frames.append(annotated)

        return output_frames