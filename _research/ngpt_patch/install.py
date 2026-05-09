"""Monkey-patch entry point for nGPT tiers.

Wired from ``pretrain_gpt.py`` next to the logging-patch install. No-op
unless one of the ``APERTUS_NGPT_*`` env vars is set.

T1 only for now: row-unit-norm projection after each train step (and
once at model build, so iter 0 forward already sees normalized weights).
"""

from __future__ import annotations

import os

from . import projection

_INSTALLED = False


def _enabled_t1() -> bool:
    return os.environ.get("APERTUS_NGPT_WEIGHT_PROJECTION") == "1"


def _enabled_any() -> bool:
    return _enabled_t1()


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not _enabled_any():
        return
    _INSTALLED = True

    if _enabled_t1():
        _install_t1_projection()


def _install_t1_projection() -> None:
    """Wrap setup_model_and_optimizer + train_step to project weights."""
    from megatron.training import training as mtt

    original_setup = mtt.setup_model_and_optimizer

    def wrapped_setup(*args, **kwargs):
        result = original_setup(*args, **kwargs)
        # ``result`` is (model, optimizer, opt_param_scheduler) — model is a list of chunks.
        model = result[0] if isinstance(result, tuple) else result
        projection.project_unit_row(model)
        return result

    mtt.setup_model_and_optimizer = wrapped_setup

    original_train_step = mtt.train_step

    def wrapped_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        # train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config)
        model = kwargs.get("model")
        if model is None and len(args) >= 3:
            model = args[2]
        if model is not None:
            projection.project_unit_row(model)
        return result

    mtt.train_step = wrapped_train_step
