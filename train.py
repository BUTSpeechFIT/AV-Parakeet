# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from utils.nemo import allow_external_nemo_targets

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNOAV

allow_external_nemo_targets()


def maybe_load_pretrained_model(model_name: str | None) -> ASRModel | None:
    if not model_name:
        return None
    return ASRModel.from_pretrained(model_name=model_name, map_location="cpu")


@hydra_runner(config_path="conf", config_name="av_parakeet")
def main(cfg):
    logging.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    init_from_pretrained = cfg.get("init_from_pretrained")
    pretrained_model = maybe_load_pretrained_model(init_from_pretrained)

    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTBPEModelSTNOAV(
        cfg=cfg.model,
        trainer=trainer,
        tokenizer=pretrained_model.tokenizer if pretrained_model is not None else None,
    )

    if pretrained_model is not None:
        missing, unexpected = asr_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        logging.info("Missing keys: %s", missing)
        logging.info("Unexpected keys: %s", unexpected)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    if cfg.get("decode_only", False):
        trainer.validate(asr_model)
        return

    if cfg.get("evaluate_at_start", True):
        trainer.validate(asr_model)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
