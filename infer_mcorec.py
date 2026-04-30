from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.core.config import hydra_runner
from nemo.utils import logging

from utils.inference import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_AVHUBERT_CHUNK_SIZE,
    InferenceRuntime,
    decode_video_frames,
    extract_word_spans,
    format_vtt_timestamp,
    frame_range_from_metadata,
    frames_to_video_tensor,
    load_json,
    load_audio_and_num_frames,
)


FPS = 25


def load_crop_metadata(session_dir: Path, track: dict[str, Any]) -> dict[str, Any]:
    metadata_path = session_dir / track.get("crop_metadata", "")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Crop metadata not found: {metadata_path}")

    return load_json(metadata_path)


def create_filled_lip_frames(
    session_dir: Path,
    tracks: list[dict[str, Any]],
    total_frames: int,
    fps: int = FPS,
) -> np.ndarray | None:
    """
    Builds a full-session lip-video timeline.

    Frames covered by a speaker crop are filled with that crop.
    Frames without a crop are black.
    """

    track_segments = []
    target_height = None
    target_width = None

    for track in tracks:
        relative_path = track.get("lip") or track.get("video")
        if not relative_path:
            continue

        video_path = session_dir / relative_path
        if not video_path.exists():
            continue

        metadata = load_crop_metadata(session_dir, track)
        start_frame, end_frame = frame_range_from_metadata(metadata, fps)

        frames = decode_video_frames(video_path)
        if len(frames) == 0:
            continue

        height, width = frames[0].shape[:2]
        target_height = height if target_height is None else max(target_height, height)
        target_width = width if target_width is None else max(target_width, width)

        track_segments.append(
            {
                "start": start_frame,
                "end": end_frame,
                "frames": frames,
            }
        )

    if not track_segments or target_height is None or target_width is None:
        return None

    track_segments.sort(key=lambda segment: segment["start"])

    output_frames = []

    for frame_index in range(total_frames):
        frame = None

        for segment in track_segments:
            if segment["start"] <= frame_index < segment["end"]:
                local_index = frame_index - segment["start"]
                if local_index < len(segment["frames"]):
                    frame = segment["frames"][local_index]
                break

        if frame is None:
            frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(
                frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        output_frames.append(frame)

    return np.stack(output_frames, axis=0)


class SessionSpeakerDataset(Dataset):
    def __init__(
        self,
        session_dirs: list[str | Path],
        audio_featurizer,
        video_transform,
        fps: int = FPS,
        view: str = "central",
    ):
        self.session_dirs = [Path(path) for path in session_dirs]
        self.audio_featurizer = audio_featurizer
        self.video_transform = video_transform
        self.fps = fps
        self.view = view
        self.items = self._build_items()
        self.audio_cache = {}

    def _build_items(self) -> list[dict[str, Any]]:
        items = []

        for session_path in self.session_dirs:
            metadata_path = session_path / "metadata.json"
            if not metadata_path.exists():
                logging.warning(f"Skipping session without metadata.json: {session_path}")
                continue

            metadata = load_json(metadata_path)

            for speaker_name, speaker_data in metadata.items():
                items.append(
                    {
                        "session_path": session_path,
                        "session_name": session_path.name,
                        "speaker_name": speaker_name,
                        "speaker_data": speaker_data,
                    }
                )

        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_session_audio(
        self,
        session_path: Path,
        video_relative_path: str,
    ) -> tuple[torch.Tensor, int, int]:
        session_key = str(session_path)

        if session_key in self.audio_cache:
            cached = self.audio_cache[session_key]
            return cached["audio"], cached["sample_rate"], cached["total_frames"]

        video_path = session_path / video_relative_path
        if not video_path.exists():
            raise FileNotFoundError(f"Session video not found: {video_path}")

        audio, sample_rate, total_frames = load_audio_and_num_frames(
            video_path,
            self.audio_featurizer,
        )

        self.audio_cache[session_key] = {
            "audio": audio,
            "sample_rate": sample_rate,
            "total_frames": total_frames,
        }

        cached = self.audio_cache[session_key]
        return cached["audio"], cached["sample_rate"], cached["total_frames"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]

        session_path = item["session_path"]
        session_name = item["session_name"]
        speaker_name = item["speaker_name"]
        speaker_data = item["speaker_data"]

        view_data = speaker_data.get(self.view, {})
        crops = view_data.get("crops", [])
        video_relative_path = view_data.get("video")

        if not video_relative_path:
            return self._error(item, "Missing session video field.")

        if not crops:
            return self._error(item, "No speaker crops found.")

        try:
            audio, sample_rate, total_frames = self._load_session_audio(
                session_path,
                video_relative_path,
            )

            uem = view_data.get("uem", {})
            uem_start = float(uem.get("start", 0.0))
            uem_end = float(uem.get("end", total_frames / self.fps))

            start_audio_index = int(uem_start * sample_rate)
            end_audio_index = int(uem_end * sample_rate)
            start_video_index = int(uem_start * self.fps)
            end_video_index = int(uem_end * self.fps)

            lip_frames = create_filled_lip_frames(
                session_dir=session_path,
                tracks=crops,
                total_frames=total_frames,
                fps=self.fps,
            )

            if lip_frames is None:
                return self._error(item, "Failed to create filled lip frames.")

            video_tensor = frames_to_video_tensor(
                lip_frames,
                self.video_transform,
                add_speaker_dim=True,
            )

            return {
                "success": True,
                "session_name": session_name,
                "speaker_name": speaker_name,
                "audio": audio[start_audio_index:end_audio_index],
                "video": video_tensor[start_video_index:end_video_index],
                "uem_start": uem_start,
                "uem_end": uem_end,
            }

        except Exception as error:
            return self._error(item, str(error))

    @staticmethod
    def _error(item: dict[str, Any], message: str) -> dict[str, Any]:
        return {
            "success": False,
            "session_name": item["session_name"],
            "speaker_name": item["speaker_name"],
            "error": message,
        }


def collate_items(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class MCoReCInferenceEngine:
    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.cfg = cfg
        self.fps = cfg.get("fps", FPS)
        self.view = cfg.get("view", "central")

        logging.info("Loading ASR model...")
        self.runtime = InferenceRuntime(cfg, device_name=device)
        logging.info("ASR model loaded successfully.")

    def transcribe(self, audio: torch.Tensor, video: torch.Tensor):
        return self.runtime.transcribe(
            audio,
            video,
            timestamps=self.cfg.get("timestamps", False),
            avhubert_chunk_size=int(
                self.cfg.get("avhubert_chunk_size", DEFAULT_AVHUBERT_CHUNK_SIZE)
            ),
        )

    def process_sessions(
        self,
        session_dirs: list[str | Path],
        output_dir: str | Path,
        num_workers: int = 8,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = SessionSpeakerDataset(
            session_dirs=session_dirs,
            audio_featurizer=self.runtime.audio_featurizer,
            video_transform=self.runtime.video_transform,
            fps=self.fps,
            view=self.view,
        )

        if len(dataset) == 0:
            logging.warning("No speakers found.")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=collate_items,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        logging.info(
            f"Processing {len(dataset)} speaker(s) across "
            f"{len(session_dirs)} session(s)."
        )

        results_by_session = {}

        for batch in tqdm(dataloader, desc="Processing speakers", unit="speaker"):
            item = batch[0]

            if not item["success"]:
                logging.warning(
                    f"[{item['session_name']}/{item['speaker_name']}] "
                    f"{item.get('error', 'Unknown error')}"
                )
                continue

            transcript = self.transcribe(item["audio"], item["video"])

            session_name = item["session_name"]
            speaker_name = item["speaker_name"]

            results_by_session.setdefault(session_name, {})[speaker_name] = {
                "text": transcript.text,
                "uem_start": item["uem_start"],
                "uem_end": item["uem_end"],
                "word_spans": (
                    extract_word_spans(
                        transcript,
                        self.runtime.token_duration_seconds,
                        start_offset_seconds=item["uem_start"],
                    )
                    if self.cfg.get("timestamps", False)
                    and hasattr(transcript, "timestamp")
                    else None
                ),
            }

        self._write_outputs(results_by_session, output_dir)

    def _write_outputs(
        self,
        results_by_session: dict[str, dict[str, dict[str, Any]]],
        output_dir: Path,
    ) -> None:
        for session_name, speaker_results in results_by_session.items():
            session_output_dir = output_dir / session_name
            session_output_dir.mkdir(parents=True, exist_ok=True)

            for speaker_name, transcript in speaker_results.items():
                output_path = session_output_dir / f"{speaker_name}.vtt"
                self._write_vtt(output_path, transcript)
                logging.info(f"Saved transcript: {output_path}")

    def _write_vtt(self, output_path: Path, transcript: dict[str, Any]) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            if self.cfg.get("timestamps", False) and transcript.get("word_spans") is not None:
                self._write_word_level_vtt(f, transcript)
            else:
                self._write_segment_vtt(f, transcript)

    @staticmethod
    def _write_segment_vtt(file, transcript: dict[str, Any]) -> None:
        text = transcript["text"].strip().replace("<unk>", "").strip()
        if not text:
            return

        start = format_vtt_timestamp(transcript["uem_start"] + 0.5)
        end = format_vtt_timestamp(transcript["uem_end"] - 0.5)

        file.write(f"{start} --> {end}\n{text}\n\n")

    @staticmethod
    def _write_word_level_vtt(file, transcript: dict[str, Any]) -> None:
        for span in transcript["word_spans"]:
            start = format_vtt_timestamp(span.start)
            end = format_vtt_timestamp(span.end)
            file.write(f"{start} --> {end}\n{span.text}\n\n")


@hydra_runner(config_path="conf", config_name="av_parakeet")
def main(cfg: DictConfig) -> None:
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    session_dir = cfg.get("session_dir")
    output_dir = cfg.get("output_dir")
    num_workers = cfg.get("num_workers", 8)

    if not output_dir:
        raise ValueError("output_dir must be specified.")

    engine = MCoReCInferenceEngine(cfg)

    if not session_dir:
        logging.info("No session_dir specified. Model loaded successfully.")
        return

    session_dirs = (
        sorted(glob.glob(session_dir))
        if str(session_dir).strip().endswith("*")
        else [session_dir]
    )

    logging.info(f"Processing {len(session_dirs)} session(s).")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"DataLoader workers: {num_workers}")

    engine.process_sessions(
        session_dirs=session_dirs,
        output_dir=output_dir,
        num_workers=num_workers,
    )

    logging.info("Completed processing all sessions.")


if __name__ == "__main__":
    main()  # noqa: pylint: disable=no-value-for-parameter
