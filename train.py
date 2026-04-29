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

# PREPEND PATH
import os
import sys
sys.path.append(os.path.dirname(__file__))

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.collections.asr.models import ASRModel

# We need to change the behavior of the _is_target_allowed function to be able to specify targets outside of nemo.
import nemo.core.classes
nemo.core.classes.common._is_target_allowed = lambda x: True

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNOAV

from pytorch_lightning.callbacks import Callback
class EvalAtStartCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Evaluating at start...")
        trainer.validate(pl_module)

@hydra_runner(config_path="conf", config_name="av_parakeet")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    init_from_pretrained = cfg.get("init_from_pretrained", None)
    pretrained_model = None
    if init_from_pretrained is not None:
        pretrained_model = ASRModel.from_pretrained(model_name=init_from_pretrained, map_location='cpu')

    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecRNNTBPEModelSTNOAV(cfg=cfg.model, trainer=trainer, tokenizer=pretrained_model.tokenizer if pretrained_model is not None else None)

    if init_from_pretrained is not None:
        missing, unexpected = asr_model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

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
