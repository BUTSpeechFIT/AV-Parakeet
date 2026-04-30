from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.inference import (
    TranscriptSpan,
    collect_video_paths,
    format_vtt_timestamp,
    frame_range_from_metadata,
    merge_spans,
    write_ctm,
)


class InferenceHelperTests(unittest.TestCase):
    def test_merge_spans_merges_adjacent_words_within_gap(self):
        spans = [
            TranscriptSpan(text="hello", start=0.0, end=0.4),
            TranscriptSpan(text="world", start=0.45, end=0.9),
            TranscriptSpan(text="again", start=1.5, end=1.8),
        ]

        merged = merge_spans(spans, max_gap_seconds=0.1)

        self.assertEqual(
            merged,
            [
                TranscriptSpan(text="hello world", start=0.0, end=0.9),
                TranscriptSpan(text="again", start=1.5, end=1.8),
            ],
        )

    def test_collect_video_paths_filters_and_sorts_supported_files(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            avi_path = root / "a.avi"
            mp4_path = root / "b.mp4"
            txt_path = root / "notes.txt"
            avi_path.write_bytes(b"")
            mp4_path.write_bytes(b"")
            txt_path.write_text("ignore me", encoding="utf-8")

            video_paths = collect_video_paths(root)

            self.assertEqual(video_paths, [avi_path, mp4_path])

    def test_collect_video_paths_rejects_duplicate_stems(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "clip.mp4").write_bytes(b"")
            (root / "clip.mkv").write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "Duplicate video stems"):
                collect_video_paths(root)

    def test_frame_range_from_metadata_prefers_explicit_frame_values(self):
        metadata = {"frame_start": 10, "frame_end": 42, "start_time": 2.0, "end_time": 3.0}
        self.assertEqual(frame_range_from_metadata(metadata, fps=25), (10, 42))

    def test_frame_range_from_metadata_falls_back_to_times(self):
        metadata = {"start_time": 1.2, "end_time": 2.8}
        self.assertEqual(frame_range_from_metadata(metadata, fps=10), (12, 28))

    def test_format_vtt_timestamp_clamps_negative_values(self):
        self.assertEqual(format_vtt_timestamp(-1.0), "00:00:00.000")
        self.assertEqual(format_vtt_timestamp(3661.234), "01:01:01.233")

    def test_write_ctm_serializes_word_spans(self):
        spans = [TranscriptSpan(text="hello", start=1.0, end=1.4)]

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.ctm"
            write_ctm(output_path, "utt-1", spans)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "utt-1 1 1.000 0.400 hello\n",
            )


if __name__ == "__main__":
    unittest.main()
