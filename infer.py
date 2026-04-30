from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.inference import (
    DEFAULT_AVHUBERT_CHUNK_SIZE,
    InferenceRuntime,
    collect_video_paths,
    extract_word_spans,
    load_audio_video_tensors,
    merge_spans,
    write_ctm,
)


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


def infer_to_ctm(
    runtime: InferenceRuntime,
    video_path: Path,
    output_dir: Path,
    merge_gap_seconds: float | None,
    avhubert_chunk_size: int,
) -> Path:
    audio, video = load_audio_video_tensors(
        video_path,
        runtime.audio_featurizer,
        runtime.video_transform,
    )
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
    runtime = InferenceRuntime.from_checkpoint(checkpoint_path, device_name=args.device)

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
