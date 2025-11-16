"""
Robust Homography helper.

- Computes homography (optionally with RANSAC).
- Transforms arrays of 2D points or single 2D points.
- Provides inverse transform and access to raw matrix and mask.
"""

from typing import Optional, Tuple, Union
import numpy as np
import cv2


ArrayLike = Union[np.ndarray, list, tuple]


def _to_numpy_points(points: ArrayLike) -> np.ndarray:
    """
    Convert input (Nx2) array-like to numpy float32 array of shape (N,2).
    Accepts lists/tuples or numpy arrays. Empty input returns array with shape (0,2).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(-1, 2)
    if pts.ndim == 1:
        # maybe a single 2-element point
        if pts.size != 2:
            raise ValueError("1D points must have length 2 (x,y).")
        pts = pts.reshape(1, 2)
    if pts.ndim == 2 and pts.shape[1] == 2:
        return pts
    # try to reshape if user passed nested lists like [[x,y],[x,y],...]
    try:
        return pts.reshape(-1, 2)
    except Exception:
        raise ValueError("Points must be convertible to shape (N,2).")


class Homography:
    def __init__(
        self,
        src_points: ArrayLike,
        dst_points: ArrayLike,
        *,
        use_ransac: bool = True,
        ransac_reproj_threshold: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.995,
    ) -> None:
        """
        Compute homography mapping src_points -> dst_points.

        Args:
            src_points: iterable of shape (N,2) (x,y) source coordinates.
            dst_points: iterable of shape (N,2) (x,y) target coordinates.
            use_ransac: if True, use cv2.findHomography with RANSAC to be robust to outliers.
            ransac_reproj_threshold: reprojection threshold for RANSAC.
            max_iters: max iterations for RANSAC (passed via cv2 flag if available).
            confidence: RANSAC confidence.
        Raises:
            ValueError if shapes mismatch or homography cannot be computed.
        """
        src = _to_numpy_points(src_points)
        dst = _to_numpy_points(dst_points)

        if src.shape[0] < 4 or dst.shape[0] < 4:
            # homography generally needs >=4 point correspondences for robust solution (unless you explicitly allow affine)
            raise ValueError("At least 4 point correspondences are required to compute a homography.")

        if src.shape != dst.shape:
            raise ValueError(f"Source and target must have same shape; got {src.shape} vs {dst.shape}.")

        # attempt to compute
        if use_ransac:
            # OpenCV's findHomography supports RANSAC; newer versions accept ransacReprojThreshold and optional flags.
            H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=float(ransac_reproj_threshold),
                                         maxIters=int(max_iters),
                                         confidence=float(confidence))
        else:
            H, mask = cv2.findHomography(src, dst, method=0)  # 0 == regular method

        if H is None:
            raise ValueError("Homography matrix could not be estimated (cv2.findHomography returned None).")

        self._H = H.astype(np.float64)
        # mask might be None for non-ransac; normalize to Nx1 uint8 when present
        self._mask = None if mask is None else mask.astype(np.uint8).reshape(-1)

        # Precompute inverse if invertible
        try:
            self._H_inv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError:
            self._H_inv = None

    @property
    def matrix(self) -> np.ndarray:
        """Return homography matrix (3x3) as numpy array (dtype float64)."""
        return self._H

    @property
    def mask(self) -> Optional[np.ndarray]:
        """Return RANSAC mask (1D uint8 array) if computed, else None."""
        return self._mask

    def transform_points(self, points: ArrayLike) -> np.ndarray:
        """
        Transform N x 2 points using the computed homography.
        Returns an (N,2) float32 numpy array.

        Accepts:
            - list/tuple/numpy array of points shape (N,2)
            - single point as (x,y) or [x,y] (will return shape (1,2))

        Returns:
            numpy.ndarray shape (N,2)
        """
        pts = _to_numpy_points(points)
        if pts.size == 0:
            return pts.reshape(-1, 2).astype(np.float32)

        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts_reshaped, self._H)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_point(self, point: ArrayLike) -> Tuple[float, float]:
        """Transform a single (x,y) point and return (x', y')."""
        pts = _to_numpy_points(point)
        if pts.shape[0] != 1:
            raise ValueError("transform_point expects a single 2D point.")
        t = self.transform_points(pts)
        return float(t[0, 0]), float(t[0, 1])

    def inverse_transform_points(self, points: ArrayLike) -> np.ndarray:
        """
        Transform points using the inverse homography (maps dst -> src).
        Raises if inverse could not be computed.
        """
        if self._H_inv is None:
            raise ValueError("Inverse homography matrix is not available (homography not invertible).")

        pts = _to_numpy_points(points)
        if pts.size == 0:
            return pts.reshape(-1, 2).astype(np.float32)

        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts_reshaped, self._H_inv)
        return transformed.reshape(-1, 2).astype(np.float32)

    def inverse_transform_point(self, point: ArrayLike) -> Tuple[float, float]:
        pts = _to_numpy_points(point)
        if pts.shape[0] != 1:
            raise ValueError("inverse_transform_point expects a single 2D point.")
        t = self.inverse_transform_points(pts)
        return float(t[0, 0]), float(t[0, 1])

    def __repr__(self) -> str:
        return f"<Homography matrix shape={self._H.shape} invertible={self._H_inv is not None}>"