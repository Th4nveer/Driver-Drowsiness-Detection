import numpy as np
from scipy.spatial import distance as dist

# Mouth landmark indices
TOP = 13
BOTTOM = 14
LEFT = 78
RIGHT = 308


def calculate_mar(landmarks, frame_width, frame_height):
    """
    Calculates Mouth Aspect Ratio (MAR).

    Parameters:
        landmarks     : MediaPipe facial landmarks
        frame_width   : Width of video frame
        frame_height  : Height of video frame

    Returns:
        float : MAR value
    """

    # Convert normalized coordinates to pixel coordinates
    top = (
        landmarks[TOP].x * frame_width,
        landmarks[TOP].y * frame_height
    )

    bottom = (
        landmarks[BOTTOM].x * frame_width,
        landmarks[BOTTOM].y * frame_height
    )

    left = (
        landmarks[LEFT].x * frame_width,
        landmarks[LEFT].y * frame_height
    )

    right = (
        landmarks[RIGHT].x * frame_width,
        landmarks[RIGHT].y * frame_height
    )

    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)

    # Prevent division by zero
    if horizontal == 0:
        return 0.0

    mar = vertical / horizontal

    return mar