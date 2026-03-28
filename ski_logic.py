"""
Video I/O, YOLO pose estimation, and Gemini coaching.

Pure kinematics and run summaries live in `ski_analysis`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from ski_analysis import (
    DEFAULT_SEGMENT_WEIGHTS,
    analyze_frame,
    center_of_mass,
    edge_angulation,
    posture_heuristics,
    smooth_keypoint_trajectories,
    summarize_run_data,
)
from ski_gemini import GeminiSkiCoach

__all__ = [
    "SkiVideoProcessor",
    "GeminiSkiCoach",
    "summarize_run_data",
]


def _ffmpeg_executable() -> str | None:
    """System ``ffmpeg`` on PATH, else the binary shipped with ``imageio-ffmpeg``."""
    w = shutil.which("ffmpeg")
    if w:
        return w
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _reencode_mp4_for_browser(path: str) -> bool:
    """Replace *path* with H.264 / yuv420p MP4 in-place when any ffmpeg is available."""
    if not path:
        return False
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        return False
    abs_path = os.path.abspath(path)
    out_dir = os.path.dirname(abs_path) or "."
    fd, tmp = tempfile.mkstemp(suffix=".mp4", dir=out_dir)
    os.close(fd)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                abs_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                tmp,
            ],
            check=True,
            timeout=900,
        )
        os.replace(tmp, abs_path)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


class SkiVideoProcessor:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        *,
        model: Any | None = None,
        segment_weights: dict[str, float] | None = None,
    ):
        self.model = model if model is not None else YOLO(model_path)
        self.segment_weights = (
            dict(segment_weights) if segment_weights is not None else dict(DEFAULT_SEGMENT_WEIGHTS)
        )

    def process_video(self, input_path: str, output_path: str) -> list[dict[str, Any]]:
        print("Pass 1: Extracting raw keypoints...")
        raw_kps, fps, width, height = self._extract_raw_keypoints(input_path)

        if len(raw_kps) == 0:
            raise ValueError("No people detected in the video.")

        print("Pass 2: Smoothing trajectories...")
        raw_arr = np.array(raw_kps)
        smoothed_kps = smooth_keypoint_trajectories(raw_arr)

        print("Pass 3: Analyzing posture and rendering output...")
        run_data = self._analyze_and_render(
            input_path, output_path, raw_arr, smoothed_kps, fps, width, height
        )

        print(f"Video processing complete. Saved to {output_path}")
        return run_data

    def _extract_raw_keypoints(
        self, video_path: str
    ) -> tuple[list[np.ndarray], float, int, int]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        raw_kps: list[np.ndarray] = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = self.model(frame, verbose=False)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                raw_kps.append(results[0].keypoints.xy[0].cpu().numpy())
            else:
                raw_kps.append(np.zeros((17, 2)))
        cap.release()
        return raw_kps, fps, width, height

    def _calculate_center_of_mass(self, kp: np.ndarray) -> tuple[float, float]:
        return center_of_mass(kp, self.segment_weights)

    def _calculate_posture_heuristics(
        self, kp: np.ndarray, com_x: float, com_y: float
    ) -> dict[str, Any]:
        return posture_heuristics(kp, com_x, com_y)

    def _calculate_edge_angulation(self, kp: np.ndarray) -> float:
        return edge_angulation(kp)

    def _analyze_frame(
        self, kp: np.ndarray, com_x: float, com_y: float
    ) -> dict[str, Any] | None:
        return analyze_frame(kp, com_x, com_y)

    def _draw_skeleton_and_com(
        self, frame: np.ndarray, kp: np.ndarray, com_x: float, com_y: float
    ) -> np.ndarray:
        edges = [
            (15, 13),
            (13, 11),
            (16, 14),
            (14, 12),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
        ]
        for e in edges:
            pt1 = (int(kp[e[0]][0]), int(kp[e[0]][1]))
            pt2 = (int(kp[e[1]][0]), int(kp[e[1]][1]))
            if pt1 != (0, 0) and pt2 != (0, 0):
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        for j in kp:
            if (int(j[0]), int(j[1])) != (0, 0):
                cv2.circle(frame, (int(j[0]), int(j[1])), 4, (0, 255, 0), -1)
        if com_x != 0 and com_y != 0:
            cv2.circle(frame, (int(com_x), int(com_y)), 10, (255, 0, 0), -1)
        return frame

    def _analyze_and_render(
        self,
        input_path: str,
        output_path: str,
        raw_kps: np.ndarray,
        smoothed_kps: np.ndarray,
        fps: float,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(input_path)
        # mp4v = MPEG-4 Part 2; fine for VLC/download, often won't play in browser <video>.
        # For reliable in-browser preview, re-encode to H.264 (e.g. ffmpeg) or use an H.264 fourcc if your OpenCV build supports it.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        run_data: list[dict[str, Any]] = []

        n_frames = min(len(raw_kps), len(smoothed_kps))
        while cap.isOpened() and frame_idx < n_frames:
            success, frame = cap.read()
            if not success:
                break

            sm_kp = smoothed_kps[frame_idx]
            com_x, com_y = self._calculate_center_of_mass(sm_kp)
            frame_analysis = self._analyze_frame(sm_kp, com_x, com_y)

            if frame_analysis:
                run_data.append(
                    {"frame": frame_idx, "com": (com_x, com_y), **frame_analysis}
                )
                raw_kp = raw_kps[frame_idx]
                kp_draw = raw_kp if np.any(raw_kp != 0) else sm_kp
                com_dx, com_dy = self._calculate_center_of_mass(kp_draw)
                frame = self._draw_skeleton_and_com(frame, kp_draw, com_dx, com_dy)

                is_profile = frame_analysis["posture"]["is_profile_view"]
                ratio = frame_analysis["posture"]["view_ratio"]

                view_text = "View: Profile" if is_profile else "View: Front/Back"
                view_color = (0, 255, 0) if is_profile else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"{view_text} (Ratio: {ratio})",
                    (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    view_color,
                    2,
                )

                cv2.putText(
                    frame,
                    f"Edge Angle: {frame_analysis['edge_inclination_deg']} deg",
                    (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                if "BACKSEAT" in frame_analysis["flags"] and is_profile:
                    cv2.putText(
                        frame,
                        "BACKSEAT DETECTED",
                        (40, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        if _reencode_mp4_for_browser(output_path):
            print("Re-encoded output to H.264 for browser playback (ffmpeg).")

        return run_data
