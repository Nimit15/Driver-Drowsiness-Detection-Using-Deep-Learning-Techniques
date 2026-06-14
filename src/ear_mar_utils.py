# ear_mar_utils.py
# Utility functions for Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR)
# and drowsiness score computation
# Written as part of NTCC project — Driver Drowsiness Detection

from scipy.spatial import distance as dist
import numpy as np


# EAR thresholds — tuned based on testing
EAR_THRESHOLD  = 0.25   # below this = eyes closing
MAR_THRESHOLD  = 0.50   # above this = yawning
TILT_THRESHOLD = 20.0   # degrees, above this = head tilting


def calculate_EAR(eye_landmarks):
    """
    Calculates Eye Aspect Ratio from 6 eye landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    When eye is open  -> EAR is roughly 0.25 to 0.35
    When eye is closed -> EAR drops below 0.20

    Args:
        eye_landmarks: list of 6 (x, y) tuples — eye landmark coordinates

    Returns:
        float: EAR value rounded to 4 decimal places
    """
    # vertical distances
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # horizontal distance
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    ear = (A + B) / (2.0 * C)
    return round(ear, 4)


def calculate_MAR(mouth_landmarks):
    """
    Calculates Mouth Aspect Ratio from 8 mouth landmark points.

    Higher MAR means mouth is more open (yawning).
    Normal closed mouth -> MAR around 0.2 to 0.3
    Yawning -> MAR rises above 0.5

    Args:
        mouth_landmarks: list of 8 (x, y) tuples — mouth landmark coordinates

    Returns:
        float: MAR value rounded to 4 decimal places
    """
    # vertical distances (3 pairs across the mouth)
    A = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
    B = dist.euclidean(mouth_landmarks[2], mouth_landmarks[6])
    C = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
    # horizontal distance
    D = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])

    mar = (A + B + C) / (3.0 * D)
    return round(mar, 4)


def drowsiness_score(ear, mar, head_tilt_angle,
                     ear_thresh=EAR_THRESHOLD,
                     mar_thresh=MAR_THRESHOLD,
                     tilt_thresh=TILT_THRESHOLD):
    """
    Combines EAR, MAR and head tilt into a single drowsiness score (0 to 1).

    Weights:
        EAR signal   -> 40%
        MAR signal   -> 30%
        Head tilt    -> 30%

    Score of 0.0 = fully awake
    Score of 1.0 = fully drowsy

    Args:
        ear: float — current Eye Aspect Ratio
        mar: float — current Mouth Aspect Ratio
        head_tilt_angle: float — head tilt in degrees
        ear_thresh: float — threshold below which eyes are considered closing
        mar_thresh: float — threshold above which mouth is considered open (yawn)
        tilt_thresh: float — threshold above which head tilt is considered abnormal

    Returns:
        float: drowsiness score between 0.0 and 1.0
    """
    # normalise each signal to 0-1 range (clamped)
    ear_signal  = max(0.0, (ear_thresh - ear) / ear_thresh)
    mar_signal  = max(0.0, (mar - mar_thresh) / (1.0 - mar_thresh))
    tilt_signal = max(0.0, (head_tilt_angle - tilt_thresh) / (90.0 - tilt_thresh))

    score = (0.40 * ear_signal) + (0.30 * mar_signal) + (0.30 * tilt_signal)
    return round(min(score, 1.0), 4)


def is_drowsy(score, threshold=0.50):
    """
    Returns True if the drowsiness score exceeds the threshold.

    Args:
        score: float — output from drowsiness_score()
        threshold: float — cutoff above which driver is classified as drowsy

    Returns:
        bool
    """
    return score >= threshold


def get_status_label(score):
    """
    Returns a human-readable status string based on the drowsiness score.

    Args:
        score: float — 0.0 to 1.0

    Returns:
        str: 'AWAKE', 'ALERT', or 'DROWSY'
    """
    if score < 0.30:
        return 'AWAKE'
    elif score < 0.50:
        return 'ALERT'
    else:
        return 'DROWSY'


# quick test when run directly
if __name__ == '__main__':
    # test EAR
    open_eye   = [(0,0),(1,2),(2,2),(4,0),(3,2),(1,2)]
    closed_eye = [(0,0),(1,0.2),(2,0.2),(4,0),(3,0.2),(1,0.2)]
    print('EAR open   :', calculate_EAR(open_eye))
    print('EAR closed :', calculate_EAR(closed_eye))

    # test MAR
    closed_mouth = [(0,0),(1,0.3),(2,0.3),(3,0.3),(6,0),(5,0.3),(4,0.3),(3,0.3)]
    open_mouth   = [(0,0),(1,1.5),(2,1.5),(3,1.5),(6,0),(5,1.5),(4,1.5),(3,1.5)]
    print('MAR closed :', calculate_MAR(closed_mouth))
    print('MAR open   :', calculate_MAR(open_mouth))

    # test score
    print()
    print('Score (awake)  :', drowsiness_score(0.30, 0.25, 5))
    print('Score (eyes)   :', drowsiness_score(0.15, 0.25, 5))
    print('Score (yawn)   :', drowsiness_score(0.28, 0.70, 5))
    print('Score (tilt)   :', drowsiness_score(0.28, 0.25, 40))
    print('Score (drowsy) :', drowsiness_score(0.10, 0.80, 50))
