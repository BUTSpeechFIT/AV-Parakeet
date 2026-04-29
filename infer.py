from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils.inference import (
    DEFAULT_AVHUBERT_CHUNK_SIZE,
    InferenceRuntime,
    TranscriptSpan,
    decode_video_frames,
    extract_word_spans,
    frames_to_video_tensor,
    load_audio_from_video,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "conf" / "av_parakeet.yaml"
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AV speech recognition on one video or a directory of videos.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="Path to a single video or to a directory of videos.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where one CTM file per video will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.ckpt or .ptl).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on. Use "auto", "cpu", "cuda", or "cuda:N".',
    )
    parser.add_argument(
        "--avhubert-chunk-size",
        type=int,
        default=DEFAULT_AVHUBERT_CHUNK_SIZE,
        help="Chunk size used by the visual encoder during inference.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Skip failed videos and keep processing the rest.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=None,
        help=(
            "If set, merge adjacent words into one CTM segment whenever the gap "
            "between them is at most this many seconds."
        ),
    )
    args = parser.parse_args()

    if args.avhubert_chunk_size <= 0:
        parser.error("--avhubert-chunk-size must be greater than 0.")
    if args.merge_gap_seconds is not None and args.merge_gap_seconds < 0:
        parser.error("--merge-gap-seconds must be greater than or equal to 0.")

    return args


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def load_config(checkpoint_path: Path) -> DictConfig:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {DEFAULT_CONFIG_PATH}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    cfg.init_from_ptl_ckpt = str(checkpoint_path)
    return cfg


def merge_spans(
    spans: list[TranscriptSpan],
    max_gap_seconds: float | None,
) -> list[TranscriptSpan]:
    if max_gap_seconds is None or not spans:
        return spans

    merged: list[TranscriptSpan] = []
    current_text = [spans[0].text]
    current_start = spans[0].start
    current_end = spans[0].end

    for span in spans[1:]:
        if span.start - current_end <= max_gap_seconds:
            current_text.append(span.text)
            current_end = max(current_end, span.end)
            continue

        merged.append(
            TranscriptSpan(
                text=" ".join(current_text),
                start=current_start,
                end=current_end,
            )
        )
        current_text = [span.text]
        current_start = span.start
        current_end = span.end

    merged.append(
        TranscriptSpan(
            text=" ".join(current_text),
            start=current_start,
            end=current_end,
        )
    )
    return merged


def write_ctm(output_path: Path, utterance_id: str, spans: list[TranscriptSpan]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for span in spans:
            duration = max(0.0, span.end - span.start)
            file_obj.write(
                f"{utterance_id} 1 {span.start:.3f} {duration:.3f} {span.text}\n"
            )


def collect_video_paths(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    video_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise ValueError(f"No supported video files found in: {input_path}")

    duplicate_stems = sorted(
        stem for stem, count in Counter(path.stem for path in video_paths).items()
        if count > 1
    )
    if duplicate_stems:
        duplicates = ", ".join(duplicate_stems)
        raise ValueError(
            "Duplicate video stems would overwrite CTM outputs: "
            f"{duplicates}"
        )

    return video_paths


def infer_to_ctm(
    runtime: InferenceRuntime,
    video_path: Path,
    output_dir: Path,
    merge_gap_seconds: float | None,
    avhubert_chunk_size: int,
) -> Path:
    audio = load_audio_from_video(video_path, runtime.audio_featurizer)
    frames = decode_video_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    video = frames_to_video_tensor(frames, runtime.video_transform)
    transcript = runtime.transcribe(
        audio,
        video,
        timestamps=True,
        avhubert_chunk_size=avhubert_chunk_size,
    )
    spans = extract_word_spans(transcript, runtime.token_duration_seconds)
    spans = merge_spans(spans, merge_gap_seconds)

    output_path = output_dir / f"{video_path.stem}.ctm"
    write_ctm(output_path, video_path.stem, spans)
    return output_path


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    video_paths = collect_video_paths(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s", checkpoint_path)
    runtime = InferenceRuntime(
        cfg=load_config(checkpoint_path),
        device_name=args.device,
    )

    logging.info("Processing %d video(s)", len(video_paths))
    failures: list[tuple[Path, Exception]] = []
    successes = 0

    for video_path in video_paths:
        try:
            output_path = infer_to_ctm(
                runtime=runtime,
                video_path=video_path,
                output_dir=output_dir,
                merge_gap_seconds=args.merge_gap_seconds,
                avhubert_chunk_size=args.avhubert_chunk_size,
            )
        except Exception as error:
            if not args.continue_on_fail:
                raise
            failures.append((video_path, error))
            logging.error("Failed: %s (%s)", video_path, error)
            continue

        successes += 1
        logging.info("Saved %s", output_path)

    if failures:
        logging.error(
            "Completed with %d success(es) and %d failure(s).",
            successes,
            len(failures),
        )
        return 1

    logging.info("Completed successfully. Wrote %d CTM file(s).", successes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
