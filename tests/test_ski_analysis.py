"""Unit tests for pure kinematics and run summarization (ski_analysis)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from ski_analysis import (
    DEFAULT_SEGMENT_WEIGHTS,
    analyze_frame,
    center_of_mass,
    edge_angulation,
    posture_heuristics,
    smooth_keypoint_trajectories,
    summarize_run_data,
)


class TestSmoothKeypointTrajectories(unittest.TestCase):
    def test_short_sequence_returns_copy_without_crash(self):
        raw = np.zeros((3, 17, 2))
        raw[:, 0, 0] = [1.0, 2.0, 3.0]
        out = smooth_keypoint_trajectories(raw, window_length=11, polyorder=3)
        self.assertEqual(out.shape, raw.shape)

    def test_long_sequence_finite_same_shape(self):
        t = np.linspace(0, 4 * math.pi, 40)
        raw = np.zeros((len(t), 17, 2))
        raw[:, 5, 0] = np.sin(t) * 100 + 50
        raw[:, 5, 1] = np.cos(t) * 100 + 50
        out = smooth_keypoint_trajectories(raw, window_length=11, polyorder=3)
        self.assertEqual(out.shape, raw.shape)
        self.assertTrue(np.all(np.isfinite(out)))


class TestCenterOfMass(unittest.TestCase):
    def test_all_zero_returns_origin(self):
        kp = np.zeros((17, 2))
        self.assertEqual(center_of_mass(kp), (0.0, 0.0))

    def test_single_point_head_weighted(self):
        kp = np.zeros((17, 2))
        kp[0] = [100.0, 200.0]
        cx, cy = center_of_mass(kp, DEFAULT_SEGMENT_WEIGHTS)
        w = DEFAULT_SEGMENT_WEIGHTS["head"]
        self.assertAlmostEqual(cx, 100.0 * w, places=5)
        self.assertAlmostEqual(cy, 200.0 * w, places=5)


class TestPostureHeuristics(unittest.TestCase):
    def test_empty_legs_returns_defaults(self):
        kp = np.zeros((17, 2))
        kp[5] = [1, 1]
        kp[6] = [2, 1]
        h = posture_heuristics(kp, 0.0, 0.0)
        self.assertFalse(h["is_profile_view"])
        self.assertEqual(h["knee_angle"], 0.0)


class TestEdgeAngulation(unittest.TestCase):
    def test_zero_keypoints_returns_zero(self):
        kp = np.zeros((17, 2))
        self.assertEqual(edge_angulation(kp), 0.0)

    def test_horizontal_knee_ankle_offset_returns_ninety(self):
        # dy == 0 branch in edge helper => 90° (leg segment treated as horizontal in image plane)
        kp = np.zeros((17, 2))
        kp[13] = [100.0, 100.0]
        kp[15] = [200.0, 100.0]
        kp[14] = [80.0, 100.0]
        kp[16] = [20.0, 100.0]
        ang = edge_angulation(kp)
        self.assertEqual(ang, 90.0)


class TestAnalyzeFrame(unittest.TestCase):
    def test_all_zero_returns_none(self):
        kp = np.zeros((17, 2))
        self.assertIsNone(analyze_frame(kp, 0.0, 0.0))


class TestSummarizeRunData(unittest.TestCase):
    def test_empty_run_returns_none(self):
        self.assertIsNone(summarize_run_data([]))

    def test_backseat_and_waist_percentages(self):
        frames = [
            {
                "flags": ["BACKSEAT", "BREAKING AT WAIST"],
                "edge_inclination_deg": 10.0,
            },
            {"flags": [], "edge_inclination_deg": 5.0},
            {"flags": ["BACKSEAT"], "edge_inclination_deg": 20.0},
        ]
        s = summarize_run_data(frames)
        assert s is not None
        self.assertEqual(s["duration_frames"], 3)
        self.assertAlmostEqual(s["backseat_percentage"], round(2 / 3 * 100, 1))
        self.assertAlmostEqual(s["breaking_at_waist_percentage"], round(1 / 3 * 100, 1))

    def test_carving_score_from_active_edges(self):
        frames = [{"flags": [], "edge_inclination_deg": 30.0} for _ in range(5)]
        s = summarize_run_data(frames)
        assert s is not None
        self.assertEqual(s["carving_score"], 50)
        self.assertEqual(s["average_active_edge_deg"], 30.0)

    def test_no_active_turns_zero_carving_score(self):
        frames = [{"flags": [], "edge_inclination_deg": 10.0} for _ in range(3)]
        s = summarize_run_data(frames)
        assert s is not None
        self.assertEqual(s["carving_score"], 0)
        self.assertEqual(s["average_active_edge_deg"], 0.0)

    def test_carving_score_capped_at_100(self):
        frames = [{"flags": [], "edge_inclination_deg": 72.0} for _ in range(4)]
        s = summarize_run_data(frames)
        assert s is not None
        self.assertEqual(s["carving_score"], 100)

    def test_max_edge_tracks_frames(self):
        frames = [
            {"flags": [], "edge_inclination_deg": 12.0},
            {"flags": [], "edge_inclination_deg": 48.0},
        ]
        s = summarize_run_data(frames)
        assert s is not None
        self.assertEqual(s["max_edge_inclination_deg"], 48.0)


class TestAnalyzeFrameBranches(unittest.TestCase):
    def test_profile_backseat_sets_flag(self):
        kp = np.zeros((17, 2))
        # Narrow width / tall torso => profile view (ratio < 0.35)
        kp[5] = [100.0, 40.0]
        kp[6] = [103.0, 40.0]
        kp[11] = [100.0, 200.0]
        kp[12] = [103.0, 200.0]
        kp[13] = [100.0, 280.0]
        kp[14] = [200.0, 280.0]
        kp[15] = [100.0, 380.0]
        kp[16] = [200.0, 380.0]
        com_x, com_y = 50.0, 250.0
        out = analyze_frame(kp, com_x, com_y)
        assert out is not None
        self.assertIn("BACKSEAT", out["flags"])
        self.assertTrue(out["posture"]["is_profile_view"])

    def test_stiff_legs_flag(self):
        kp = np.zeros((17, 2))
        kp[5] = [100.0, 40.0]
        kp[6] = [130.0, 40.0]
        kp[11] = [100.0, 200.0]
        kp[12] = [130.0, 200.0]
        hip = np.array([115.0, 200.0])
        knee = np.array([115.0, 320.0])
        ankle = np.array([115.0, 400.0])
        kp[13] = knee
        kp[15] = ankle
        kp[14] = knee
        kp[16] = ankle
        com_x, com_y = center_of_mass(kp)
        out = analyze_frame(kp, com_x, com_y)
        assert out is not None
        self.assertIn("STIFF LEGS", out["flags"])


class TestSmoothDefaultWindow(unittest.TestCase):
    def test_default_window_runs(self):
        raw = np.zeros((20, 17, 2))
        raw[:, 3, 0] = np.linspace(0, 19, 20)
        out = smooth_keypoint_trajectories(raw)
        self.assertEqual(out.shape, raw.shape)


if __name__ == "__main__":
    unittest.main()
