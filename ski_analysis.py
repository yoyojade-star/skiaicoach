"""
Pure kinematics and run summarization (no OpenCV / YOLO / network).
Used by SkiVideoProcessor and unit tests.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

DEFAULT_SEGMENT_WEIGHTS: dict[str, float] = {
    "head": 0.081,
    "trunk": 0.497,
    "upper_arm_l": 0.028,
    "upper_arm_r": 0.028,
    "forearm_l": 0.022,
    "forearm_r": 0.022,
    "thigh_l": 0.100,
    "thigh_r": 0.100,
    "calf_l": 0.061,
    "calf_r": 0.061,
}


def smooth_keypoint_trajectories(
    raw_kps: np.ndarray, window_length: int = 7, polyorder: int = 3
) -> np.ndarray:
    """Applies Savitzky–Golay smoothing to each keypoint's x and y coordinates.

    This function smooths the temporal trajectory of each of the 17 keypoints
    independently. It is used to reduce jitter from frame-to-frame predictions.
    The window length is dynamically adjusted for short sequences to prevent errors.

    Args:
        raw_kps (np.ndarray): A NumPy array of shape (T, 17, 2) containing the
            raw keypoint coordinates, where T is the number of frames.
        window_length (int, optional): The length of the filter window. Must be
            a positive odd integer. Defaults to 7.
        polyorder (int, optional): The order of the polynomial used to fit the
            samples. Must be less than window_length. Defaults to 3.

    Returns:
        np.ndarray: A NumPy array of the same shape as `raw_kps` with the
            smoothed keypoint coordinates.
    """
    num_frames = raw_kps.shape[0]
    smoothed_kps = np.copy(raw_kps)
    wl = window_length
    if num_frames < wl:
        wl = num_frames if num_frames % 2 != 0 else num_frames - 1
        if wl <= polyorder:
            return smoothed_kps

    for k in range(17):
        x, y = raw_kps[:, k, 0], raw_kps[:, k, 1]
        if np.count_nonzero(x) > wl:
            smoothed_kps[:, k, 0] = savgol_filter(x, wl, polyorder)
            smoothed_kps[:, k, 1] = savgol_filter(y, wl, polyorder)
    return smoothed_kps


def center_of_mass(
    kp: np.ndarray, segment_weights: dict[str, float] | None = None
) -> tuple[float, float]:
    """Calculates the center of mass for a single frame of keypoints.

    The center of mass (COM) is computed as a weighted average of the centers
    of predefined body segments. If all keypoints are zero, the COM is (0, 0).

    Args:
        kp (np.ndarray): A NumPy array of shape (17, 2) for a single frame,
            representing the (x, y) coordinates of 17 COCO-style keypoints.
        segment_weights (dict[str, float] | None, optional): A dictionary
            mapping body segments to their proportional weight. If None,
            `DEFAULT_SEGMENT_WEIGHTS` is used.

    Returns:
        tuple[float, float]: A tuple containing the (x, y) coordinates of the
            calculated center of mass.
    """
    if np.all(kp == 0):
        return 0.0, 0.0
    w = segment_weights or DEFAULT_SEGMENT_WEIGHTS
    segments = {
        "head": kp[0],
        "trunk": (((kp[5] + kp[6]) / 2) + ((kp[11] + kp[12]) / 2)) / 2,
        "upper_arm_l": (kp[5] + kp[7]) / 2,
        "upper_arm_r": (kp[6] + kp[8]) / 2,
        "forearm_l": (kp[7] + kp[9]) / 2,
        "forearm_r": (kp[8] + kp[10]) / 2,
        "thigh_l": (kp[11] + kp[13]) / 2,
        "thigh_r": (kp[12] + kp[14]) / 2,
        "calf_l": (kp[13] + kp[15]) / 2,
        "calf_r": (kp[14] + kp[16]) / 2,
    }
    com_x = sum(segments[p][0] * w[p] for p in w)
    com_y = sum(segments[p][1] * w[p] for p in w)
    return float(com_x), float(com_y)


def posture_heuristics(
    kp: np.ndarray, com_x: float, _com_y: float
) -> dict[str, Any]:
    """Analyzes a single frame of keypoints to determine posture characteristics.

    This function calculates various metrics and flags related to skiing posture,
    such as being "backseat," "breaking at the waist," and having "stiff legs."
    It also computes hip and knee angles and determines if the skier is in a
    profile view relative to the camera.

    Args:
        kp (np.ndarray): A NumPy array of shape (17, 2) for a single frame.
        com_x (float): The x-coordinate of the center of mass for the frame.
        _com_y (float): The y-coordinate of the center of mass (unused).

    Returns:
        dict[str, Any]: A dictionary containing various posture flags and angles.
            Keys include 'is_backseat', 'breaking_at_waist', 'stiff_legs',
            'hip_angle', 'knee_angle', 'is_profile_view', and 'view_ratio'.
    """
    h: dict[str, Any] = {
        "is_backseat": False,
        "breaking_at_waist": False,
        "stiff_legs": False,
        "hip_angle": 0.0,
        "knee_angle": 0.0,
        "is_profile_view": False,
        "view_ratio": 0.0,
    }

    avg_knee_x = max(kp[13][0], kp[14][0])
    avg_hip_x = max(kp[11][0], kp[12][0])
    if avg_knee_x == 0 or avg_hip_x == 0:
        return h

    direction = "right" if avg_knee_x > avg_hip_x else "left"

    if direction == "right":
        shoulder, hip, knee, ankle = kp[6], kp[12], kp[14], kp[16]
    else:
        shoulder, hip, knee, ankle = kp[5], kp[11], kp[13], kp[15]

    if (
        np.all(shoulder == 0)
        or np.all(hip == 0)
        or np.all(knee == 0)
        or np.all(ankle == 0)
    ):
        return h

    shoulder_width = abs(kp[5][0] - kp[6][0])
    hip_width = abs(kp[11][0] - kp[12][0])
    torso_height = abs(kp[5][1] - kp[11][1])

    if torso_height > 0:
        avg_body_width = (shoulder_width + hip_width) / 2
        view_ratio = avg_body_width / torso_height
        h["view_ratio"] = round(float(view_ratio), 2)
        if view_ratio < 0.35:
            h["is_profile_view"] = True

    if h["is_profile_view"]:
        leg_length = abs(hip[1] - ankle[1])
        tolerance = leg_length * 0.10
        if direction == "right":
            if com_x < (ankle[0] - tolerance):
                h["is_backseat"] = True
        else:
            if com_x > (ankle[0] + tolerance):
                h["is_backseat"] = True

    v_thigh = np.array([hip[0] - knee[0], hip[1] - knee[1]])
    v_calf = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    if np.linalg.norm(v_thigh) > 0 and np.linalg.norm(v_calf) > 0:
        cos_knee = np.dot(v_thigh, v_calf) / (
            np.linalg.norm(v_thigh) * np.linalg.norm(v_calf)
        )
        h["knee_angle"] = math.degrees(math.acos(np.clip(cos_knee, -1.0, 1.0)))

    v_torso = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
    v_femur = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    if np.linalg.norm(v_torso) > 0 and np.linalg.norm(v_femur) > 0:
        cos_hip = np.dot(v_torso, v_femur) / (
            np.linalg.norm(v_torso) * np.linalg.norm(v_femur)
        )
        h["hip_angle"] = math.degrees(math.acos(np.clip(cos_hip, -1.0, 1.0)))

    if h["knee_angle"] > 160:
        h["stiff_legs"] = True
    if h["hip_angle"] < 90 and h["knee_angle"] > 140:
        h["breaking_at_waist"] = True

    return h


def edge_angulation(kp: np.ndarray) -> float:
    """Calculates the average ski edge inclination from ankle-knee geometry.

    This function estimates the ski edge angle relative to the vertical axis
    by calculating the angle of the lower leg (calf) for both legs and
    averaging the results.

    Args:
        kp (np.ndarray): A NumPy array of shape (17, 2) for a single frame.

    Returns:
        float: The average inclination angle in degrees. Returns 0.0 if no
            valid leg keypoints are found.
    """

    def calc_inc(k: np.ndarray, a: np.ndarray) -> float | None:
        if np.all(k == 0) or np.all(a == 0):
            return None
        dx, dy = abs(k[0] - a[0]), abs(a[1] - k[1])
        return 90.0 if dy == 0 else math.degrees(math.atan(dx / dy))

    angles = [
        a
        for a in (calc_inc(kp[13], kp[15]), calc_inc(kp[14], kp[16]))
        if a is not None
    ]
    return round(sum(angles) / len(angles), 1) if angles else 0.0


def analyze_frame(
    kp: np.ndarray, com_x: float, com_y: float
) -> dict[str, Any] | None:
    """Performs a complete analysis for a single frame.

    This function combines posture heuristics and edge angulation calculations
    to create a comprehensive analysis record for one frame of video.

    Args:
        kp (np.ndarray): A NumPy array of shape (17, 2) for a single frame.
        com_x (float): The x-coordinate of the center of mass for the frame.
        com_y (float): The y-coordinate of the center of mass for the frame.

    Returns:
        dict[str, Any] | None: A dictionary containing the analysis results,
            including posture data, edge inclination, and a list of flags.
            Returns None if the keypoint data for the frame is all zero.
    """
    if np.all(kp == 0):
        return None
    posture = posture_heuristics(kp, com_x, com_y)
    edge = edge_angulation(kp)
    flags: list[str] = []
    if posture.get("is_backseat"):
        flags.append("BACKSEAT")
    if posture.get("breaking_at_waist"):
        flags.append("BREAKING AT WAIST")
    if posture.get("stiff_legs"):
        flags.append("STIFF LEGS")
    return {"posture": posture, "edge_inclination_deg": edge, "flags": flags}


def summarize_run_data(run_data: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregates per-frame analysis data into a run summary.

    This function takes a list of frame-by-frame analysis results and computes
    summary statistics for the entire ski run, such as the percentage of time
    spent in poor posture, maximum edge angles, and a carving score.

    Args:
        run_data (list[dict[str, Any]]): A list of dictionaries, where each
            dictionary is the output of `analyze_frame` for a single frame.

    Returns:
        dict[str, Any] | None: A dictionary containing summary metrics for the
            run. Returns None if the input `run_data` is empty.
    """
    total_frames = len(run_data)
    if total_frames == 0:
        return None

    backseat_frames = sum(1 for f in run_data if "BACKSEAT" in f.get("flags", []))
    waist_frames = sum(
        1 for f in run_data if "BREAKING AT WAIST" in f.get("flags", [])
    )

    edge_angles = [f.get("edge_inclination_deg", 0) for f in run_data]
    max_edge = max(edge_angles, default=0)

    active_turns = [angle for angle in edge_angles if angle > 15.0]
    if active_turns:
        avg_active_edge = sum(active_turns) / len(active_turns)
        carving_score = min(int((avg_active_edge / 60.0) * 100), 100)
    else:
        avg_active_edge = 0.0
        carving_score = 0

    return {
        "duration_frames": total_frames,
        "backseat_percentage": round((backseat_frames / total_frames) * 100, 1),
        "breaking_at_waist_percentage": round((waist_frames / total_frames) * 100, 1),
        "max_edge_inclination_deg": max_edge,
        "average_active_edge_deg": round(avg_active_edge, 1),
        "carving_score": carving_score,
    }