#!/usr/bin/env python3
"""Create Lhotse CutSet manifests from an LRS2 directory tree."""

import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

from lhotse import AudioSource, Recording, SupervisionSegment
from lhotse.audio.utils import VideoInfo
from lhotse.cut import CutSet, MonoCut
from torchcodec.decoders import AudioDecoder, VideoDecoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_video_files(dataset_dir: Path) -> list[Path]:
    """Return all `.video` files under `dataset_dir` in deterministic order."""
    video_files: list[Path] = []
    for root, dirs, files in os.walk(dataset_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith(".video"):
                video_files.append(Path(root) / filename)
    return video_files


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and strip leading/trailing whitespace."""
    return path.read_text(encoding="utf-8").strip()


def extract_speaker_id(sample_id: str) -> str:
    """Extract the speaker id from an LRS2 sample id."""
    parts = sample_id.split("/")
    if len(parts) >= 2 and parts[1]:
        return parts[1]

    if "portrait_face" in sample_id:
        speaker_id = sample_id.split("_portrait_face_", maxsplit=1)[0]
        if speaker_id:
            return speaker_id

    raise ValueError(f"Invalid sample ID format: {sample_id!r}")


def create_cut_from_video(video_path: Path, dataset_part: str) -> Optional[MonoCut]:
    """Create a single Lhotse cut from one `.video` file."""
    try:
        label_path = video_path.with_suffix(".label")
        sample_id_path = video_path.with_suffix(".sample_id")

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found for {video_path}")
        if not sample_id_path.exists():
            raise FileNotFoundError(f"Sample ID file not found for {video_path}")

        transcript = read_text_file(label_path)
        sample_id = read_text_file(sample_id_path)
        if not sample_id:
            raise ValueError(f"Sample ID file is empty for {video_path}")

        speaker_id = extract_speaker_id(sample_id)
        cut_id = f"{dataset_part}_{sample_id.replace('/', '_')}"

        audio_decoder = AudioDecoder(str(video_path))
        video_decoder = VideoDecoder(str(video_path))
        audio_samples = audio_decoder.get_all_samples()
        duration = audio_samples.duration_seconds

        recording = Recording(
            id=cut_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=str(video_path.resolve()),
                    video=VideoInfo(
                        fps=int(round(video_decoder.metadata.average_fps)),
                        num_frames=video_decoder.metadata.num_frames,
                        height=video_decoder.metadata.height,
                        width=video_decoder.metadata.width,
                    ),
                )
            ],
            sampling_rate=audio_decoder.metadata.sample_rate,
            num_samples=audio_samples.data.shape[-1],
            duration=duration,
        )

        supervision = SupervisionSegment(
            id=cut_id,
            recording_id=cut_id,
            start=0.0,
            duration=duration,
            channel=0,
            text=transcript,
            language="en",
            speaker=speaker_id,
        )

        return MonoCut(
            id=cut_id,
            start=0.0,
            duration=duration,
            channel=0,
            supervisions=[supervision],
            recording=recording,
            custom={
                "dataset_part": dataset_part,
                "sample_id": sample_id,
                "per_spk_lip_crop_videos": {speaker_id: str(video_path.resolve())},
            },
        )
    except Exception as exc:
        logging.error("Error processing %s: %s", video_path, exc)
        return None


def process_single_video(video_path: Path, part_name: str) -> Optional[MonoCut]:
    """Helper for per-file processing, including multiprocessing."""
    return create_cut_from_video(video_path, part_name)


def process_dataset_part(dataset_dir: Path, part_name: str, num_workers: int = 1) -> CutSet:
    """Process one dataset partition into a Lhotse CutSet."""
    logging.info("Processing dataset part: %s", part_name)

    video_files = find_video_files(dataset_dir)
    logging.info("Found %d video files", len(video_files))

    cuts: list[MonoCut] = []
    if num_workers == 1:
        for video_path in tqdm(video_files, desc=f"Creating cuts for {part_name}"):
            cut = process_single_video(video_path, part_name)
            if cut is not None:
                cuts.append(cut)
    else:
        logging.info("Using %d worker processes", num_workers)
        process_func = partial(process_single_video, part_name=part_name)
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_func, video_files),
                    total=len(video_files),
                    desc=f"Creating cuts for {part_name}",
                )
            )
        cuts = [cut for cut in results if cut is not None]
        cuts.sort(key=lambda cut: cut.id)

    logging.info("Successfully processed %d out of %d video files", len(cuts), len(video_files))

    cutset = CutSet.from_cuts(cuts)
    logging.info("Created CutSet with %d cuts", len(cutset))
    return cutset


def main() -> None:
    """Parse arguments and build LRS2 manifests."""
    parser = argparse.ArgumentParser(
        description="Create Lhotse CutSets from LRS2 dataset directory structure"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the LRS2 root directory containing dataset partitions",
    )
    parser.add_argument(
        "--output_manifest_dir",
        type=Path,
        required=True,
        help="Directory where the output CutSet manifests will be stored",
    )
    parser.add_argument(
        "--process_parts",
        type=str,
        nargs="+",
        help="Optional list of partition directory names to process",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes to use "
            "(default: 1 for sequential processing, use -1 to use all available CPU cores)"
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if the output manifest file already exists",
    )

    args = parser.parse_args()

    if args.num_workers == -1:
        args.num_workers = cpu_count()
        logging.info("Using all available CPU cores: %d", args.num_workers)
    elif args.num_workers < 1:
        raise ValueError(
            f"Number of workers must be >= 1 or -1 for all cores, got {args.num_workers}"
        )
    elif args.num_workers > 1:
        logging.info("Using %d worker processes", args.num_workers)

    if not args.data_dir.exists():
        raise ValueError(f"LRS2 directory does not exist: {args.data_dir}")
    if not args.data_dir.is_dir():
        raise ValueError(f"LRS2 path is not a directory: {args.data_dir}")

    args.output_manifest_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output manifests will be saved to: %s", args.output_manifest_dir)

    subdirs = {
        subdir.name: subdir
        for subdir in sorted(args.data_dir.iterdir(), key=lambda path: path.name)
        if subdir.is_dir()
    }
    if not subdirs:
        raise ValueError(f"No subdirectories found in {args.data_dir}")

    if args.process_parts:
        part_names = list(dict.fromkeys(args.process_parts))
        unknown_parts = sorted(name for name in part_names if name not in subdirs)
        if unknown_parts:
            raise ValueError(
                f"Unknown dataset partitions requested: {unknown_parts}. "
                f"Available partitions: {sorted(subdirs)}"
            )
    else:
        part_names = list(subdirs)

    logging.info("Processing dataset partitions: %s", part_names)

    for part_name in part_names:
        subdir = subdirs[part_name]
        output_path = args.output_manifest_dir / f"{part_name}_cuts.jsonl.gz"

        if args.skip_existing and output_path.exists():
            logging.info("Skipping %s because manifest already exists at %s", part_name, output_path)
            continue

        cutset = process_dataset_part(subdir, part_name, num_workers=args.num_workers)
        logging.info("Saving CutSet to: %s", output_path)
        cutset.to_file(output_path)
        logging.info("Successfully saved %s CutSet", part_name)

    logging.info("All dataset partitions processed successfully")


if __name__ == "__main__":
    main()
