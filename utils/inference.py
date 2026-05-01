from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils.nemo import allow_external_nemo_targets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

allow_external_nemo_targets()

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None

from src.data.transforms import VideoTransform
from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNOAV


AUDIO_SAMPLE_RATE = 16_000
DEFAULT_AVHUBERT_CHUNK_SIZE = 20
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "av_parakeet.yaml"
DEFAULT_VIDEO_FPS = 25
VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
    }
)


@dataclass(frozen=True)
class TranscriptSpan:
    text: str
    start: float
    end: float


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_inference_config(
    checkpoint_path: str | Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg.init_from_ptl_ckpt = str(checkpoint_path)
    return cfg


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
        raise ValueError(
            "Duplicate video stems would overwrite CTM outputs: "
            + ", ".join(duplicate_stems)
        )

    return video_paths


def merge_spans(
    spans: list[TranscriptSpan],
    max_gap_seconds: float | None,
) -> list[TranscriptSpan]:
    if max_gap_seconds is None or len(spans) < 2:
        return spans

    merged: list[TranscriptSpan] = []
    current = spans[0]

    for span in spans[1:]:
        if span.start - current.end <= max_gap_seconds:
            current = TranscriptSpan(
                text=f"{current.text} {span.text}",
                start=current.start,
                end=max(current.end, span.end),
            )
            continue

        merged.append(current)
        current = span

    merged.append(current)
    return merged


def write_ctm(
    output_path: Path,
    utterance_id: str,
    spans: list[TranscriptSpan],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        for span in spans:
            duration = max(0.0, span.end - span.start)
            file_obj.write(
                f"{utterance_id} 1 {span.start:.3f} {duration:.3f} {span.text}\n"
            )


def frame_range_from_metadata(
    metadata: Mapping[str, Any],
    fps: int = DEFAULT_VIDEO_FPS,
) -> tuple[int, int]:
    if "frame_start" in metadata and "frame_end" in metadata:
        return int(metadata["frame_start"]), int(metadata["frame_end"])

    start_time = float(metadata.get("start_time", 0.0))
    end_time = float(metadata.get("end_time", start_time))
    return round(start_time * fps), round(end_time * fps)


def format_vtt_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def require_video_decoder() -> None:
    if VideoDecoder is None:
        raise RuntimeError("torchcodec.VideoDecoder is required for inference.")


def create_video_decoder(video_path: Path, dimension_order: str = "NHWC") -> VideoDecoder:
    require_video_decoder()
    return VideoDecoder(
        str(video_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order=dimension_order,
    )


def decode_video_frames(video_path: Path) -> np.ndarray:
    decoder = create_video_decoder(video_path)
    if len(decoder) == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    return np.stack([decoder[index].numpy() for index in range(len(decoder))], axis=0)


def create_audio_featurizer() -> WaveformFeaturizer:
    return WaveformFeaturizer(
        sample_rate=AUDIO_SAMPLE_RATE,
        int_values=False,
        augmentor=None,
    )


def create_test_video_transform() -> VideoTransform:
    return VideoTransform(subset="test")


def load_audio_from_video(
    video_path: Path,
    audio_featurizer: WaveformFeaturizer,
) -> torch.Tensor:
    return audio_featurizer.process(
        video_path,
        offset=0,
        duration=0,
        trim=False,
        orig_sr=AUDIO_SAMPLE_RATE,
        channel_selector="average",
    )


def load_audio_and_num_frames(
    video_path: Path,
    audio_featurizer: WaveformFeaturizer,
) -> tuple[torch.Tensor, int, int]:
    decoder = create_video_decoder(video_path)
    audio = load_audio_from_video(video_path, audio_featurizer)
    return audio, AUDIO_SAMPLE_RATE, len(decoder)


def frames_to_video_tensor(
    frames: np.ndarray,
    video_transform: VideoTransform,
    add_speaker_dim: bool = True,
) -> torch.Tensor:
    if len(frames) == 0:
        raise RuntimeError("Video has no frames.")

    grayscale_frames = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames],
        axis=0,
    )
    video_tensor = video_transform(torch.from_numpy(grayscale_frames).unsqueeze(1))
    return video_tensor.unsqueeze(2) if add_speaker_dim else video_tensor


def load_audio_video_tensors(
    video_path: Path,
    audio_featurizer: WaveformFeaturizer,
    video_transform: VideoTransform,
    add_speaker_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = load_audio_from_video(video_path, audio_featurizer)
    frames = decode_video_frames(video_path)
    return audio, frames_to_video_tensor(
        frames,
        video_transform,
        add_speaker_dim=add_speaker_dim,
    )


def get_token_duration_seconds(cfg: DictConfig) -> float:
    return (
        float(cfg.model.preprocessor.window_stride)
        * int(cfg.model.encoder.subsampling_factor)
    )


def normalize_model_source(model_source: str | Path) -> Path | str:
    source_text = str(model_source).strip()
    if not source_text:
        raise ValueError("Model source must be specified.")

    local_path = Path(source_text).expanduser()
    if local_path.is_file():
        return local_path.resolve()

    return source_text


def load_asr_model(cfg: DictConfig) -> EncDecRNNTBPEModelSTNOAV:
    if cfg.get("init_from_ptl_ckpt") is None:
        model_source = cfg.get("init_from_pretrained")
    else:
        model_source = normalize_model_source(cfg.get("init_from_ptl_ckpt"))

    if isinstance(model_source, str):
        return EncDecRNNTBPEModelSTNOAV.from_pretrained(
            model_source,
            map_location="cpu",
        )

    checkpoint = torch.load(
        model_source,
        weights_only=False,
        map_location="cpu",
    )

    checkpoint_cfg = dict(checkpoint["hyper_parameters"].cfg)
    for key in (
        "train_ds",
        "validation_ds",
        "test_ds",
        "nemo_version",
        "visual_encoder_ckpt_path",
        "labels",
        "target",
        "tokenizer",
        "decoder",
        "joint",
        "decoding",
    ):
        checkpoint_cfg.pop(key, None)

    checkpoint_cfg["encoder"]["_target_"] = (
        "src.model.modules.av_fastconformer_encoder.ConformerEncoderSTNOAV"
    )
    cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(checkpoint_cfg))

    pretrained_name = cfg.get("init_from_pretrained")
    tokenizer = None
    if pretrained_name:
        tokenizer = ASRModel.from_pretrained(
            model_name=pretrained_name,
            map_location="cpu",
        ).tokenizer

    model = EncDecRNNTBPEModelSTNOAV(
        cfg=cfg.model,
        tokenizer=tokenizer,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


class InferenceRuntime:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device_name: str = "auto",
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> InferenceRuntime:
        return cls(
            cfg=load_inference_config(checkpoint_path, config_path=config_path),
            device_name=device_name,
        )

    def __init__(self, cfg: DictConfig, device_name: str = "auto"):
        self.cfg = cfg
        self.device = select_device(device_name)
        self.audio_featurizer = create_audio_featurizer()
        self.video_transform = create_test_video_transform()
        self.model = load_asr_model(cfg)

        self.model.to(self.device)
        self.model.eval()
        self.token_duration_seconds = get_token_duration_seconds(cfg)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        timestamps: bool,
        avhubert_chunk_size: int = DEFAULT_AVHUBERT_CHUNK_SIZE,
    ):
        transcript = self.model.custom_transcribe_single_utt(
            audio=audio.to(self.device),
            video=video.to(self.device),
            num_speakers=torch.tensor([1], dtype=torch.int64, device=self.device),
            timestamps=timestamps,
            avhubert_chunk_size=int(avhubert_chunk_size),
        )
        return transcript[0] if isinstance(transcript, list) else transcript


def extract_word_spans(
    transcript,
    token_duration_seconds: float,
    start_offset_seconds: float = 0.0,
) -> list[TranscriptSpan]:
    timestamps = getattr(transcript, "timestamp", None)
    if timestamps is None:
        raise RuntimeError("Model did not return timestamps required for word spans.")

    spans = []
    for word_info in timestamps.get("word", []):
        text = word_info.get("word", "").strip()
        if not text or text == "<unk>":
            continue

        start = (
            start_offset_seconds
            + float(word_info.get("start_offset", 0)) * token_duration_seconds
        )
        end = (
            start_offset_seconds
            + float(word_info.get("end_offset", 0)) * token_duration_seconds
        )
        spans.append(TranscriptSpan(text=text, start=start, end=max(start, end)))

    return spans
