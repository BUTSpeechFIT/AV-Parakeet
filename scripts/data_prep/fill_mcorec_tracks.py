#!/usr/bin/env python3
"""Fill gaps between speaker crop tracks and mux audio from `central_video.mp4`.

For each session directory containing `central_video.mp4` and `metadata.json`,
the script creates per-speaker `tracks_filled.mp4` and `tracks_filled_lip.mp4`
outputs. Frames outside crop ranges are filled with black frames so each output
has the same frame count as the central video.

Implementation details:
- `torchcodec.VideoDecoder` reads the source videos.
- OpenCV handles resizing before frame encoding.
- `ffmpeg` encodes the filled video and muxes audio from `central_video.mp4`.
- `ffprobe` verifies the final file contains an audio stream.
"""

import argparse
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

FPS = 25
DECODER_KWARGS = {
    "device": "cpu",
    "seek_mode": "exact",
    "num_ffmpeg_threads": 1,
    "dimension_order": "NHWC",
}


def ensure_external_dependencies() -> None:
    """Fail fast when ffmpeg or ffprobe are not available."""
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required external dependencies: "
            f"{', '.join(missing)}. Install them and ensure they are available on PATH."
        )


def load_crop_meta(session_dir: Path, track: dict[str, Any]) -> dict[str, Any]:
    """Load crop metadata for a single track."""
    crop_metadata = track.get("crop_metadata")
    if not crop_metadata:
        raise ValueError("Track is missing the 'crop_metadata' field")

    meta_path = session_dir / crop_metadata
    if not meta_path.exists():
        raise FileNotFoundError(f"Crop metadata not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_frame_range_from_meta(meta: dict[str, Any], fps: int = FPS) -> tuple[int, int]:
    """Return the inclusive-exclusive frame range for a crop track."""
    if "frame_start" in meta and "frame_end" in meta:
        return int(meta["frame_start"]), int(meta["frame_end"])

    start = int(round(float(meta.get("start_time", 0.0)) * fps))
    end = int(round(float(meta.get("end_time", start / fps)) * fps))
    return start, end


def verify_output_video(path: Path, expected_frames: int, check_audio: bool = False) -> None:
    """Verify frame count and optionally verify that an audio stream exists."""
    if not path.exists():
        raise RuntimeError(f"Output file does not exist: {path}")

    frame_decoder = VideoDecoder(str(path), **DECODER_KWARGS)
    frame_count = len(frame_decoder)
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch for {path}: expected {expected_frames}, got {frame_count}"
        )

    if check_audio:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=False, check=False)
        has_audio = result.returncode == 0 and bool(result.stdout.strip())
        if not has_audio:
            raise RuntimeError(f"Audio stream not found in {path}")

    print(
        f"Verification passed for {path}: {frame_count} frames"
        f"{', audio OK' if check_audio else ''}"
    )


def decode_frame(decoder: VideoDecoder, frame_index: int) -> np.ndarray:
    """Decode one frame into a NumPy array."""
    frame = decoder[frame_index]
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    return np.asarray(frame)


def write_combined_video(
    session_dir: Path,
    tracks: Sequence[dict[str, Any]],
    video_field: str,
    total_frames: int,
    out_path: Path,
    time_offset_seconds: float = 0.0,
    fps: int = FPS,
) -> None:
    """Create a filled video for one speaker and one crop field."""
    track_descriptors: list[dict[str, Any]] = []
    target_width: int | None = None
    target_height: int | None = None
    time_offset_frames = int(round(time_offset_seconds * fps))

    for track in tracks:
        relative_path = track.get(video_field) or track.get("video") or track.get("lip")
        if relative_path is None:
            continue

        video_path = session_dir / relative_path
        if not video_path.exists():
            print(f"Warning: track video not found (skipping): {video_path}")
            continue

        meta = load_crop_meta(session_dir, track)
        start_frame, end_frame = get_frame_range_from_meta(meta, fps=fps)
        start_frame += max(0, time_offset_frames)
        end_frame += max(0, time_offset_frames)

        decoder = VideoDecoder(str(video_path), **DECODER_KWARGS)
        num_frames = len(decoder)
        if num_frames > 0:
            frame0 = decode_frame(decoder, 0)
            height, width = int(frame0.shape[0]), int(frame0.shape[1])
            if target_width is None or target_height is None:
                target_width, target_height = width, height
            else:
                target_width = max(target_width, width)
                target_height = max(target_height, height)

        track_descriptors.append(
            {
                "start": start_frame,
                "end": end_frame,
                "decoder": decoder,
                "num_frames": num_frames,
            }
        )

    if target_width is None or target_height is None:
        print(f"No track videos found for field '{video_field}', skipping output {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_noaudio = out_path.with_suffix(".noaudio.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{target_width}x{target_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        "-pix_fmt",
        "rgb24",
        "-hide_banner",
        "-loglevel",
        "error",
        str(out_noaudio),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None or process.stderr is None:
        raise RuntimeError(f"Failed to open ffmpeg pipe for {out_noaudio}")

    track_descriptors.sort(key=lambda descriptor: descriptor["start"])
    for frame_id in tqdm(range(total_frames), desc=f"Writing {out_noaudio}", unit="frame"):
        lookup_frame_id = frame_id - min(0, time_offset_frames)
        chosen_frame: np.ndarray | None = None

        for descriptor in track_descriptors:
            if descriptor["start"] <= lookup_frame_id < descriptor["end"]:
                local_index = lookup_frame_id - descriptor["start"]
                if 0 <= local_index < descriptor["num_frames"]:
                    try:
                        chosen_frame = decode_frame(descriptor["decoder"], local_index)
                    except Exception:
                        chosen_frame = None
                break

        if chosen_frame is None:
            frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        else:
            frame = chosen_frame

        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        try:
            process.stdin.write(frame.tobytes())
        except Exception as exc:
            process.stdin.close()
            process.wait()
            raise RuntimeError(f"Failed while streaming frames to ffmpeg for {out_noaudio}") from exc

    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg exited with code {return_code} while writing {out_noaudio}: {stderr_text}"
        )

    print(f"Wrote combined video (no audio): {out_noaudio}")
    verify_output_video(out_noaudio, total_frames, check_audio=False)

    central_video = session_dir / "central_video.mp4"
    mux_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(out_noaudio),
        "-i",
        str(central_video),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-hide_banner",
        "-loglevel",
        "error",
        "-strict",
        "experimental",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(out_path),
    ]
    try:
        subprocess.run(mux_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while muxing audio into {out_path}") from exc

    out_noaudio.unlink(missing_ok=True)
    print(f"Attached audio from {central_video} -> {out_path}")
    verify_output_video(out_path, total_frames, check_audio=True)


def process_session(
    session_dir: Path,
    output_root: Path,
    crop_type: str = "central",
    fps: int = FPS,
) -> None:
    """Generate filled speaker videos for one session directory."""
    print(f"Processing session: {session_dir}")

    meta_file = session_dir / "metadata.json"
    if not meta_file.exists():
        print(f"metadata.json not found in {session_dir}, skipping")
        return

    with meta_file.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    central_video = session_dir / "central_video.mp4"
    if not central_video.exists():
        print(f"central_video.mp4 not found in {session_dir}, skipping")
        return

    total_frames = len(VideoDecoder(str(central_video), **DECODER_KWARGS))
    print(f" Central frames: {total_frames}")

    out_session = output_root / session_dir.name
    for speaker_name in sorted(metadata):
        speaker_metadata = metadata[speaker_name]
        speaker_output = out_session / "speakers" / speaker_name

        if (speaker_output / "tracks_filled.mp4").exists() or (speaker_output / "tracks_filled_lip.mp4").exists():
            print("skipping", speaker_output)
            continue

        time_offset = 0.0
        if crop_type == "ego":
            if "ego" not in speaker_metadata:
                raise ValueError(
                    f"Speaker {speaker_name} in {session_dir} does not contain an 'ego' section"
                )
            try:
                ego_conv_start = float(speaker_metadata["ego"]["uem"]["start"])
                central_conv_start = float(speaker_metadata["central"]["uem"]["start"])
            except KeyError as exc:
                raise ValueError(
                    f"Missing UEM metadata for speaker {speaker_name} in {session_dir}"
                ) from exc
            time_offset = central_conv_start - ego_conv_start

        crop_metadata = speaker_metadata.get(crop_type, {})
        tracks = crop_metadata.get("crops")
        if not tracks:
            print(
                f"No '{crop_type}' crops found for speaker {speaker_name} in {session_dir}, skipping"
            )
            continue

        write_combined_video(
            session_dir,
            tracks,
            "video",
            total_frames,
            speaker_output / "tracks_filled.mp4",
            time_offset_seconds=time_offset,
            fps=fps,
        )
        write_combined_video(
            session_dir,
            tracks,
            "lip",
            total_frames,
            speaker_output / "tracks_filled_lip.mp4",
            time_offset_seconds=time_offset,
            fps=fps,
        )


def main() -> None:
    """Parse arguments and create filled crop videos."""
    parser = argparse.ArgumentParser(
        description="Fill gaps between speaker track videos and mux central audio"
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="Session directory or glob pattern",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory where filled crop videos will be written",
    )
    parser.add_argument(
        "--crop_type",
        type=str,
        default="central",
        choices=["central", "ego"],
        help="Crop type to process (central or ego)",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Target frame rate for track alignment")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be a positive integer")

    ensure_external_dependencies()

    if glob.has_magic(args.session_dir):
        all_sessions = sorted(glob.glob(args.session_dir))
    else:
        all_sessions = [args.session_dir]

    if not all_sessions:
        raise FileNotFoundError(f"No sessions matched {args.session_dir!r}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for session in all_sessions:
        process_session(Path(session), output_root, crop_type=args.crop_type, fps=args.fps)


if __name__ == "__main__":
    main()
