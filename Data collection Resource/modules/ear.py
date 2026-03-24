import numpy as np
from scipy.spatial import distance as dist

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(landmarks, frame_width, frame_height):
    """
    Calculates average EAR (Eye Aspect Ratio) for both eyes.

    Parameters:
        landmarks     : MediaPipe facial landmarks
        frame_width   : Width of video frame
        frame_height  : Height of video frame

    Returns:
        float : Average EAR value
    """

    def eye_aspect_ratio(eye_points):
        points = []

        # Convert normalized coordinates to pixel coordinates
        for idx in eye_points:
            x = landmarks[idx].x * frame_width
            y = landmarks[idx].y * frame_height
            points.append((x, y))

        # Vertical distances
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])

        # Horizontal distance
        C = dist.euclidean(points[0], points[3])

        # Prevent division by zero
        if C == 0:
            return 0.0

        return (A + B) / (2.0 * C)

    left_ear = eye_aspect_ratio(LEFT_EYE)
    right_ear = eye_aspect_ratio(RIGHT_EYE)

    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear