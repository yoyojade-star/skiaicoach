"""Unit tests for SkiVideoProcessor (requires OpenCV + Ultralytics)."""
from __future__ import annotations

import builtins
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

import ski_analysis
import ski_logic
from ski_analysis import center_of_mass
from ski_logic import SkiVideoProcessor, _ffmpeg_executable, _reencode_mp4_for_browser


class TestSkiVideoProcessorWithMockModel(unittest.TestCase):
    def test_process_video_raises_when_no_detections(self):
        mock_model = MagicMock()
        mock_model.return_value = [MagicMock(keypoints=None)]

        proc = SkiVideoProcessor(model=mock_model)

        with patch.object(proc, "_extract_raw_keypoints", return_value=([], 30.0, 640, 480)):
            with self.assertRaises(ValueError) as ctx:
                proc.process_video("in.mp4", "out.mp4")
        self.assertIn("No people", str(ctx.exception))

    def test_smooth_path_uses_analysis_module(self):
        mock_model = MagicMock()
        proc = SkiVideoProcessor(model=mock_model)

        non_zero = np.ones((17, 2)) * 50
        non_zero[13, 0] = 60
        non_zero[15, 0] = 60
        non_zero[14, 0] = 40
        non_zero[16, 0] = 40

        raw = [non_zero, non_zero]

        with patch.object(proc, "_extract_raw_keypoints", return_value=(raw, 30.0, 64, 48)):
            with patch.object(proc, "_analyze_and_render", return_value=[]) as mock_render:
                proc.process_video("in.mp4", "out.mp4")
        mock_render.assert_called_once()
        args = mock_render.call_args[0]
        raw_arr, smoothed = args[2], args[3]
        self.assertEqual(raw_arr.shape, (2, 17, 2))
        self.assertEqual(smoothed.shape, (2, 17, 2))


class TestFfmpegExecutable(unittest.TestCase):
    def test_prefers_which_when_present(self):
        with patch("ski_logic.shutil.which", return_value=r"C:\bin\ffmpeg.exe"):
            self.assertEqual(_ffmpeg_executable(), r"C:\bin\ffmpeg.exe")

    def test_falls_back_to_imageio_ffmpeg(self):
        fake_im = MagicMock()
        fake_im.get_ffmpeg_exe.return_value = "/bundle/ffmpeg"
        with patch("ski_logic.shutil.which", return_value=None):
            with patch.dict(sys.modules, {"imageio_ffmpeg": fake_im}):
                self.assertEqual(_ffmpeg_executable(), "/bundle/ffmpeg")


class TestReencodeMp4ForBrowser(unittest.TestCase):
    def test_returns_false_when_ffmpeg_missing(self):
        with patch("ski_logic._ffmpeg_executable", return_value=None):
            self.assertFalse(_reencode_mp4_for_browser("any.mp4"))

    def test_returns_false_on_empty_path(self):
        self.assertFalse(_reencode_mp4_for_browser(""))

    def test_replaces_file_when_ffmpeg_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.mp4")
            with open(src, "wb") as f:
                f.write(b"opencvmpeg4")

            def fake_run(cmd: list[str], **_kwargs: object) -> None:
                out_path = cmd[-1]
                with open(out_path, "wb") as f:
                    f.write(b"h264mp4")

            with patch("ski_logic._ffmpeg_executable", return_value="ffmpeg"):
                with patch("ski_logic.subprocess.run", side_effect=fake_run):
                    self.assertTrue(_reencode_mp4_for_browser(src))

            with open(src, "rb") as f:
                self.assertEqual(f.read(), b"h264mp4")

    def test_returns_false_when_subprocess_fails(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.mp4")
            with open(src, "wb") as f:
                f.write(b"x")
            with patch("ski_logic._ffmpeg_executable", return_value="ffmpeg"):
                with patch(
                    "ski_logic.subprocess.run",
                    side_effect=subprocess.CalledProcessError(1, ["ffmpeg"]),
                ):
                    self.assertFalse(_reencode_mp4_for_browser(src))


class TestFfmpegExecutableImportError(unittest.TestCase):
    def test_returns_none_when_imageio_missing(self):
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object):
            if name == "imageio_ffmpeg":
                raise ImportError("no imageio_ffmpeg")
            return real_import(name, *args, **kwargs)

        with patch("ski_logic.shutil.which", return_value=None):
            with patch("builtins.__import__", side_effect=fake_import):
                self.assertIsNone(ski_logic._ffmpeg_executable())


class TestSkiVideoProcessorHelpersAndDraw(unittest.TestCase):
    def setUp(self):
        self.proc = SkiVideoProcessor(model=MagicMock())

    def test_delegate_methods_match_ski_analysis(self):
        kp = np.zeros((17, 2))
        kp[5] = [10.0, 20.0]
        kp[11] = [10.0, 100.0]
        kp[13] = [10.0, 150.0]
        kp[15] = [10.0, 200.0]
        kp[6] = [50.0, 20.0]
        kp[12] = [50.0, 100.0]
        kp[14] = [50.0, 150.0]
        kp[16] = [50.0, 200.0]
        com_x, com_y = 30.0, 80.0
        self.assertEqual(
            self.proc._calculate_center_of_mass(kp),
            center_of_mass(kp, self.proc.segment_weights),
        )
        self.assertEqual(
            self.proc._calculate_posture_heuristics(kp, com_x, com_y),
            ski_analysis.posture_heuristics(kp, com_x, com_y),
        )
        self.assertEqual(
            self.proc._calculate_edge_angulation(kp),
            ski_analysis.edge_angulation(kp),
        )
        self.assertEqual(
            self.proc._analyze_frame(kp, com_x, com_y),
            ski_analysis.analyze_frame(kp, com_x, com_y),
        )

    def test_draw_skeleton_marks_frame(self):
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        kp = np.zeros((17, 2))
        kp[5] = [60.0, 40.0]
        kp[6] = [70.0, 40.0]
        kp[11] = [60.0, 80.0]
        kp[12] = [70.0, 80.0]
        out = self.proc._draw_skeleton_and_com(frame, kp, 65.0, 50.0)
        self.assertIs(out, frame)
        self.assertGreater(int(frame.sum()), 0)


class TestExtractRawKeypoints(unittest.TestCase):
    def test_reads_frames_and_queries_model(self):
        mock_model = MagicMock()
        kpt = MagicMock()
        kpt.xy = [MagicMock()]
        kpt.xy[0].cpu.return_value.numpy.return_value = np.ones((17, 2)) * 3
        mock_model.return_value = [MagicMock(keypoints=kpt)]

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 64.0, 48.0]
        mock_cap.read.side_effect = [(True, frame), (False, None)]

        proc = SkiVideoProcessor(model=mock_model)
        with patch("ski_logic.cv2.VideoCapture", return_value=mock_cap):
            raw, fps, w, h = proc._extract_raw_keypoints("dummy.mp4")

        self.assertEqual(len(raw), 1)
        np.testing.assert_array_equal(raw[0], np.ones((17, 2)) * 3)
        self.assertEqual(fps, 30.0)
        self.assertEqual(w, 64)
        self.assertEqual(h, 48)
        mock_cap.release.assert_called_once()


class TestAnalyzeAndRender(unittest.TestCase):
    """Exercise video write loop with mocked cv2 I/O."""

    def _profile_kp(self) -> np.ndarray:
        kp = np.zeros((17, 2))
        kp[5] = [100.0, 40.0]
        kp[6] = [103.0, 40.0]
        kp[11] = [100.0, 200.0]
        kp[12] = [103.0, 200.0]
        kp[13] = [100.0, 280.0]
        kp[14] = [200.0, 280.0]
        kp[15] = [100.0, 380.0]
        kp[16] = [200.0, 380.0]
        return kp

    def test_writes_one_analyzed_frame(self):
        kp = self._profile_kp()
        raw = np.stack([kp])
        smoothed = raw.copy()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((48, 64, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_writer = MagicMock()

        proc = SkiVideoProcessor(model=MagicMock())
        with patch("ski_logic.cv2.VideoCapture", return_value=mock_cap):
            with patch("ski_logic.cv2.VideoWriter", return_value=mock_writer):
                with patch("ski_logic.cv2.VideoWriter_fourcc", return_value=0):
                    with patch("ski_logic._reencode_mp4_for_browser", return_value=False):
                        with tempfile.TemporaryDirectory() as d:
                            out_path = os.path.join(d, "o.mp4")
                            run_data = proc._analyze_and_render(
                                "in.mp4",
                                out_path,
                                raw,
                                smoothed,
                                30.0,
                                64,
                                48,
                            )

        self.assertEqual(len(run_data), 1)
        mock_writer.write.assert_called_once()
        mock_cap.release.assert_called_once()
        mock_writer.release.assert_called_once()

    def test_raw_zero_falls_back_to_smoothed_for_overlay(self):
        kp = self._profile_kp()
        raw = np.zeros((1, 17, 2))
        smoothed = np.stack([kp])

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((48, 64, 3), dtype=np.uint8)),
            (False, None),
        ]
        proc = SkiVideoProcessor(model=MagicMock())
        with patch("ski_logic.cv2.VideoCapture", return_value=mock_cap):
            with patch("ski_logic.cv2.VideoWriter", return_value=MagicMock()):
                with patch("ski_logic.cv2.VideoWriter_fourcc", return_value=0):
                    with patch("ski_logic._reencode_mp4_for_browser", return_value=False):
                        with tempfile.TemporaryDirectory() as d:
                            run_data = proc._analyze_and_render(
                                "in.mp4",
                                os.path.join(d, "o.mp4"),
                                raw,
                                smoothed,
                                30.0,
                                64,
                                48,
                            )
        self.assertEqual(len(run_data), 1)


if __name__ == "__main__":
    unittest.main()
