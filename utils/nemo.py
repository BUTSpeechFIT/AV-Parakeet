from __future__ import annotations


def allow_external_nemo_targets() -> None:
    # NeMo checkpoints in this repo refer to project-local targets.
    import nemo.core.classes

    nemo.core.classes.common._is_target_allowed = lambda _: True
