from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Allow restoring custom model targets from the checkpoint.
import nemo.core.classes
nemo.core.classes.common._is_target_allowed = lambda _: True

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


def require_video_decoder() -> None:
    if VideoDecoder is None:
        raise RuntimeError("torchcodec.VideoDecoder is required for inference.")


def decode_video_frames(video_path: Path) -> np.ndarray:
    require_video_decoder()

    decoder = VideoDecoder(
        str(video_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC",
    )
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
    require_video_decoder()

    decoder = VideoDecoder(
        str(video_path),
        device="cpu",
        seek_mode="exact",
        num_ffmpeg_threads=1,
        dimension_order="NHWC",
    )
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


def get_token_duration_seconds(cfg: DictConfig) -> float:
    return (
        float(cfg.model.preprocessor.window_stride)
        * int(cfg.model.encoder.subsampling_factor)
    )


def load_asr_model(cfg: DictConfig) -> EncDecRNNTBPEModelSTNOAV:
    checkpoint = torch.load(
        cfg.init_from_ptl_ckpt,
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
