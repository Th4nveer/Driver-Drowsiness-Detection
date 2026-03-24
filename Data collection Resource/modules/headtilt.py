import numpy as np
import math

# Eye outer corner landmarks (MediaPipe indices)
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263


def calculate_head_tilt(landmarks, frame_width, frame_height):
    """
    Calculates head tilt angle in degrees.
    
    Parameters:
        landmarks     : MediaPipe facial landmarks
        frame_width   : Width of video frame
        frame_height  : Height of video frame

    Returns:
        float : Head tilt angle in degrees
    """

    # Convert normalized coordinates to pixel coordinates
    left = (
        landmarks[LEFT_EYE_CORNER].x * frame_width,
        landmarks[LEFT_EYE_CORNER].y * frame_height
    )

    right = (
        landmarks[RIGHT_EYE_CORNER].x * frame_width,
        landmarks[RIGHT_EYE_CORNER].y * frame_height
    )

    dx = right[0] - left[0]
    dy = right[1] - left[1]

    if dx == 0:
        return 0.0

    angle = math.degrees(math.atan2(dy, dx))

    return angle