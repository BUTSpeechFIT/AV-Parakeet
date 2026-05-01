#!/usr/bin/env python3
"""Create Lhotse CutSet manifests for MCoRec sessions with filled crop videos.

The script walks sessions under `--orig-root`, reads per-speaker WebVTT labels,
maps speakers to their filled crop videos under `--filled-root`, and writes a
single Lhotse `CutSet` manifest to `--output-cuts`.

Each cut preserves the existing custom manifest fields:
- `per_spk_face_crop_videos`
- `per_spk_lip_crop_videos`
- `per_spk_asd`
- any user-provided visual feature keys
"""

import argparse
import json
import logging
import os
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio.source import AudioSource
from lhotse.audio.utils import VideoInfo
from torchcodec.decoders import AudioDecoder, VideoDecoder
from tqdm import tqdm
from webvtt import WebVTT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def time_to_seconds(timestamp: str) -> float:
    """Convert a WebVTT timestamp to seconds."""
    parts = [float(part) for part in timestamp.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    return parts[0]


def build_asd_candidate_paths(
    session_dir: Path,
    session_name: str,
    speaker_id: str,
    filled_asd_root: Optional[Path] = None,
) -> list[Path]:
    """Build candidate paths for a per-speaker ASD JSON file."""
    filename = "tracks_filled_asd.json"
    candidates: list[Path] = []
    if filled_asd_root is not None:
        candidates.append(filled_asd_root / session_name / "speakers" / speaker_id / filename)
    candidates.append(session_dir / "speakers" / speaker_id / filename)
    candidates.append(session_dir.parent / session_name / "speakers" / speaker_id / filename)
    return candidates


def parse_vtt_segments(vtt_path: Path) -> list[dict[str, Any]]:
    """Return caption segments as dictionaries with `start`, `end`, and `text`."""
    captions = WebVTT().read(str(vtt_path))
    segments: list[dict[str, Any]] = []

    for caption in captions:
        start = getattr(caption, "start", None)
        end = getattr(caption, "end", None)
        text = getattr(caption, "text", "")
        if start is None or end is None:
            continue

        try:
            start_seconds = time_to_seconds(start)
            end_seconds = time_to_seconds(end)
        except ValueError:
            continue

        normalized_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        segments.append(
            {"start": start_seconds, "end": end_seconds, "text": normalized_text}
        )

    return segments


def get_video_info_torchcodec(video_path: Path) -> dict[str, int]:
    """Return basic video metadata for a media file."""
    decoder = VideoDecoder(str(video_path))
    return {
        "num_frames": len(decoder),
        "fps": int(round(decoder.metadata.average_fps)),
        "height": decoder.metadata.height,
        "width": decoder.metadata.width,
    }


def get_audio_info_torchcodec(audio_path: Path) -> dict[str, Any]:
    """Return audio metadata for a media file."""
    decoder = AudioDecoder(str(audio_path))
    decoded = decoder.get_all_samples()
    return {
        "sampling_rate": decoder.metadata.sample_rate,
        "channels": decoder.metadata.num_channels,
        "num_samples": decoded.data.shape[-1],
        "duration": decoded.duration_seconds,
    }


def find_sessions(orig_root: Path) -> list[Path]:
    """Find all session directories that contain a `speakers` subdirectory."""
    sessions: list[Path] = []
    for root, dirs, _files in os.walk(orig_root):
        dirs.sort()
        if "speakers" in dirs:
            sessions.append(Path(root))
    return sessions


def load_uem_bounds(metadata_path: Path) -> tuple[float, float]:
    """Load shared UEM bounds and validate they are consistent across speakers."""
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not metadata:
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    uem_start: Optional[float] = None
    uem_end: Optional[float] = None

    for speaker_name in sorted(metadata):
        try:
            uem = metadata[speaker_name]["central"]["uem"]
            speaker_start = float(uem["start"])
            speaker_end = float(uem["end"])
        except KeyError as exc:
            raise ValueError(
                f"Missing central UEM metadata for speaker {speaker_name} in {metadata_path}"
            ) from exc

        if speaker_end <= speaker_start:
            raise ValueError(
                f"Invalid UEM bounds for speaker {speaker_name} in {metadata_path}: "
                f"start={speaker_start}, end={speaker_end}"
            )

        if uem_start is None or uem_end is None:
            uem_start = speaker_start
            uem_end = speaker_end
            continue

        if uem_start != speaker_start or uem_end != speaker_end:
            raise ValueError(
                f"Inconsistent UEM across speakers in {metadata_path}: "
                f"expected ({uem_start}, {uem_end}), got ({speaker_start}, {speaker_end}) "
                f"for speaker {speaker_name}"
            )

    return uem_start, uem_end


def _process_session(
    session_path_str: str,
    orig_root_str: str,
    filled_root_str: str,
    vis_features: Optional[dict[str, str]] = None,
    filled_asd_root_str: Optional[str] = None,
    allow_missing_asd: bool = False,
    use_uem: bool = False,
) -> dict[str, Any]:
    """Process a single session and return a serializable result payload."""
    session = Path(session_path_str)
    orig_root = Path(orig_root_str)
    filled_root = Path(filled_root_str)
    filled_asd_root = Path(filled_asd_root_str) if filled_asd_root_str else None

    try:
        rel_session = session.relative_to(orig_root)
        filled_session = filled_root / rel_session

        speakers_dir = session / "speakers"
        if not speakers_dir.exists():
            raise FileNotFoundError(f"Speakers directory not found in session {session}")

        speaker_dirs = sorted(
            (path for path in speakers_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )

        central_video_path = session / "central_video.mp4"
        central_video_info = get_video_info_torchcodec(central_video_path)
        central_audio_info = get_audio_info_torchcodec(central_video_path)

        if use_uem:
            uem_start, uem_end = load_uem_bounds(session / "metadata.json")
        else:
            uem_start = 0.0
            uem_end = float(central_audio_info["duration"])

        cut_id = session.name
        per_spk_face: dict[str, str] = {}
        per_spk_lip: dict[str, str] = {}
        per_spk_asd: dict[str, Optional[str]] = {}
        supervisions: list[dict[str, Any]] = []
        per_spk_features = {
            feature_key: {
                speaker_dir.name: str(Path(feature_root) / session.name / speaker_dir.name / "all_tracks.pt")
                for speaker_dir in speaker_dirs
            }
            for feature_key, feature_root in (vis_features or {}).items()
        }

        channel_ids = list(range(int(central_audio_info["channels"])))
        labels_dir = session / "labels"
        for spk_dir in speaker_dirs:
            speaker_name = spk_dir.name
            transcript_path = labels_dir / f"{speaker_name}.vtt"
            if not transcript_path.exists():
                raise FileNotFoundError(
                    f"Transcript file not found for speaker {speaker_name} at {transcript_path}"
                )

            filled_spk_dir = filled_session / "speakers" / speaker_name
            face_video = filled_spk_dir / "tracks_filled.mp4"
            lip_video = filled_spk_dir / "tracks_filled_lip.mp4"
            if not face_video.exists():
                raise FileNotFoundError(
                    f"Missing face video for {speaker_name} in {filled_spk_dir}"
                )
            if not lip_video.exists():
                raise FileNotFoundError(
                    f"Missing lip video for {speaker_name} in {filled_spk_dir}"
                )

            per_spk_face[speaker_name] = str(face_video.resolve())
            per_spk_lip[speaker_name] = str(lip_video.resolve())

            asd_candidates = build_asd_candidate_paths(
                session, session.name, speaker_name, filled_asd_root
            )
            asd_path = next(
                (str(candidate.resolve()) for candidate in asd_candidates if candidate.exists()),
                None,
            )
            if asd_path is None and allow_missing_asd and asd_candidates:
                asd_path = str(asd_candidates[0])
            per_spk_asd[speaker_name] = asd_path

            for segment_index, segment in enumerate(parse_vtt_segments(transcript_path)):
                seg_start = float(segment["start"])
                seg_end = float(segment["end"])

                if seg_end <= uem_start or seg_start >= uem_end:
                    warnings.warn(
                        f"Segment [{seg_start}, {seg_end}] is out of UEM bounds "
                        f"[{uem_start}, {uem_end}] for speaker {speaker_name} in {session}; skipping it."
                    )
                    continue

                if seg_start < uem_start or seg_end > uem_end:
                    warnings.warn(
                        f"Segment [{seg_start}, {seg_end}] exceeds UEM bounds "
                        f"[{uem_start}, {uem_end}] for speaker {speaker_name} in {session}; clamping it."
                    )
                    seg_start = max(seg_start, uem_start)
                    seg_end = min(seg_end, uem_end)

                seg_duration = seg_end - seg_start
                if seg_duration <= 0.0:
                    continue

                supervisions.append(
                    {
                        "id": f"{cut_id}_{speaker_name}_{segment_index}",
                        "recording_id": cut_id,
                        "start": seg_start,
                        "duration": seg_duration,
                        "channel": channel_ids,
                        "speaker": speaker_name,
                        "text": segment.get("text", ""),
                        "language": "en",
                    }
                )

        return {
            "result": {
                "cut_id": cut_id,
                "cut_start": uem_start,
                "cut_duration": uem_end - uem_start,
                "central_video_path": str(central_video_path),
                "central_video_info": central_video_info,
                "central_audio_info": central_audio_info,
                "per_spk_face": per_spk_face,
                "per_spk_lip": per_spk_lip,
                "per_spk_asd": per_spk_asd,
                "supervisions": supervisions,
                "vis_features": per_spk_features,
            }
        }
    except Exception as exc:
        return {"error": str(exc), "session": str(session)}


def build_manifests(
    orig_root: Path,
    filled_root: Path,
    num_workers: Optional[int] = None,
    vis_features: Optional[dict[str, str]] = None,
    filled_asd_root: Optional[Path] = None,
    allow_missing_asd: bool = False,
    use_uem: bool = False,
) -> CutSet:
    """Traverse sessions and build a Lhotse CutSet."""
    sessions = find_sessions(orig_root)
    if not sessions:
        raise ValueError(f"No sessions with speakers directories found under {orig_root}")

    logging.info("Found %d sessions under %s", len(sessions), orig_root)

    worker = partial(
        _process_session,
        orig_root_str=str(orig_root),
        filled_root_str=str(filled_root),
        vis_features=vis_features,
        filled_asd_root_str=str(filled_asd_root) if filled_asd_root else None,
        allow_missing_asd=allow_missing_asd,
        use_uem=use_uem,
    )

    session_paths = [str(session) for session in sessions]
    if num_workers is None:
        num_workers = min(cpu_count(), max(1, len(session_paths)))
    else:
        num_workers = max(1, min(int(num_workers), len(session_paths)))

    results: list[dict[str, Any]] = []
    if num_workers == 1:
        for session_path in tqdm(session_paths, total=len(session_paths), desc="Processing sessions"):
            results.append(worker(session_path))
    else:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, session_paths),
                total=len(session_paths),
                desc="Processing sessions",
            ):
                results.append(result)

    results.sort(
        key=lambda item: item.get("result", {}).get("cut_id", item.get("session", ""))
    )

    cuts: list[MonoCut] = []
    for result in results:
        if "error" in result:
            logging.error(
                "Error processing session: %s (session=%s)",
                result.get("error"),
                result.get("session"),
            )
            continue

        payload = result.get("result")
        if not payload:
            continue

        cut_id = payload["cut_id"]
        central_video_path = Path(payload["central_video_path"])
        central_video_info = payload["central_video_info"]
        central_audio_info = payload["central_audio_info"]

        recording = Recording(
            id=cut_id,
            channel_ids=list(range(int(central_audio_info["channels"]))),
            duration=central_audio_info["duration"],
            num_samples=central_audio_info["num_samples"],
            sampling_rate=central_audio_info["sampling_rate"],
            sources=[
                AudioSource(
                    channels=list(range(int(central_audio_info["channels"]))),
                    source=str(central_video_path),
                    type="file",
                    video=VideoInfo(
                        fps=int(central_video_info["fps"]),
                        height=central_video_info["height"],
                        num_frames=central_video_info["num_frames"],
                        width=central_video_info["width"],
                    ),
                )
            ],
        )

        supervisions = [
            SupervisionSegment(
                id=supervision["id"],
                recording_id=supervision["recording_id"],
                start=supervision["start"],
                duration=supervision["duration"],
                channel=supervision["channel"],
                speaker=supervision["speaker"],
                text=supervision.get("text", ""),
                language=supervision.get("language", "en"),
            )
            for supervision in payload["supervisions"]
        ]

        custom_fields = {
            "per_spk_face_crop_videos": payload["per_spk_face"],
            "per_spk_lip_crop_videos": payload["per_spk_lip"],
            "per_spk_asd": payload["per_spk_asd"],
            **payload.get("vis_features", {}),
        }

        cuts.append(
            MonoCut(
                id=f"{cut_id}_cut0",
                start=payload["cut_start"],
                duration=payload["cut_duration"],
                channel=recording.channel_ids,
                recording=recording,
                supervisions=supervisions,
                custom=custom_fields,
            )
        )

    cuts.sort(key=lambda cut: cut.id)
    return CutSet.from_cuts(cuts)


def main() -> None:
    """Parse arguments and create the MCoRec CutSet manifest."""
    parser = argparse.ArgumentParser(
        description="Create a Lhotse CutSet manifest for MCoRec filled-in crops"
    )
    parser.add_argument(
        "--orig-root",
        required=True,
        type=Path,
        help="Original MCoRec root (for example: /.../mcorec_data/dev)",
    )
    parser.add_argument(
        "--filled-root",
        required=True,
        type=Path,
        help="Filled-in crops root that mirrors --orig-root",
    )
    parser.add_argument(
        "--output-cuts",
        required=True,
        type=Path,
        help="Output Lhotse CutSet manifest path (for example: cuts.jsonl.gz)",
    )
    parser.add_argument(
        "--visual-feature-keys",
        type=str,
        nargs="+",
        default=None,
        help="Custom field keys to use for per-speaker visual feature paths",
    )
    parser.add_argument(
        "--visual-feature-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Directories containing visual features corresponding to --visual-feature-keys",
    )
    parser.add_argument(
        "--filled-asd-root",
        type=Path,
        default=None,
        help="Optional root containing filled ASD JSON outputs",
    )
    parser.add_argument(
        "--allow-missing-asd",
        action="store_true",
        help="Allow missing ASD files and still write the expected path",
    )
    parser.add_argument(
        "--use-uem",
        action="store_true",
        help="Trim the cut to the shared UEM interval defined in metadata.json",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to use",
    )

    args = parser.parse_args()

    visual_feature_keys = args.visual_feature_keys or []
    visual_feature_dirs = args.visual_feature_dirs or []
    if len(visual_feature_keys) != len(visual_feature_dirs):
        parser.error(
            "--visual-feature-keys and --visual-feature-dirs must have the same number of values"
        )

    if not args.orig_root.exists() or not args.orig_root.is_dir():
        parser.error(f"--orig-root must be an existing directory: {args.orig_root}")
    if not args.filled_root.exists() or not args.filled_root.is_dir():
        parser.error(f"--filled-root must be an existing directory: {args.filled_root}")
    if args.num_workers is not None and args.num_workers < 1:
        parser.error("--num-workers must be at least 1 when provided")

    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)

    cuts_set = build_manifests(
        args.orig_root,
        args.filled_root,
        num_workers=args.num_workers,
        vis_features=dict(zip(visual_feature_keys, visual_feature_dirs)),
        filled_asd_root=args.filled_asd_root,
        allow_missing_asd=args.allow_missing_asd,
        use_uem=args.use_uem,
    )

    logging.info("Writing cuts to %s", args.output_cuts)
    cuts_set.to_file(str(args.output_cuts))


if __name__ == "__main__":
    main()
