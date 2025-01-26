#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import subprocess
import os
import sys
from argparse import Namespace
from collections import defaultdict
from scipy import signal
from tqdm import tqdm

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar

from models.modules.losses import SISNRLoss
from utils.sound_scp import SoundScpWriter

def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    wav_writer = SoundScpWriter(f"{cfg.common_eval.results_path}/wavs", f"{cfg.common_eval.results_path}/spk.scp")
    si_snr_all = []
    dis = task.cfg.sample_rate * task.cfg.segment_shift
    sisnr_loss = SISNRLoss(1.0e-08)

    # Initialize generator
    assert cfg.common_eval.results_path is not None
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        audio_len = sample['net_input']['mix_speech_len'][0]
        mix_speech = sample['net_input']['mix_speech']
        tgt_speech = sample['net_input']['tgt_speech']

        waves = []
        for start in range(0, audio_len, dis):
            new_mix_speech = mix_speech.new_zeros((1, dis))
            new_tgt_speech = tgt_speech.new_zeros((1, dis))

            chunk_size = min(dis, audio_len - start)

            new_mix_speech[0, :chunk_size] = mix_speech[0, start:start + chunk_size]
            new_tgt_speech[0, :chunk_size] = tgt_speech[0, start:start + chunk_size]

            seg_sample = {}
            seg_sample['net_input'] = {
                "mix_speech": new_mix_speech,
                "mix_speech_len": torch.tensor([dis], dtype=torch.long),
                "tgt_speech": new_tgt_speech,
                "aux_speech": sample['net_input']['aux_speech'],
                "aux_speech_len": sample['net_input']['aux_speech_len'],
                "spk_id": sample['net_input']['spk_id'],
            }

            _, results = task.inference_step(
                models,
                seg_sample,
            )
            waves.append(results['wave'][:, :chunk_size])

        # assume batch is always 1
        wave = torch.cat(waves, dim=1)
        si_snr_all.append(sisnr_loss(tgt_speech, wave).item())
        if torch.min(wave.max(dim=1).values) > 0:
            wave = (wave / abs(wave).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
        else:
            wave = wave.cpu().numpy()
        wav_writer[f"{sample['utt_id'][0]}"] = task.cfg.sample_rate, wave[0]

    logger.info(f'SI SNR is {- sum(si_snr_all) / len(si_snr_all)}')

def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
