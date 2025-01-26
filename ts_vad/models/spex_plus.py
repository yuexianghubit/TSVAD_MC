# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import sys
import contextlib
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.models import BaseFairseqModel, register_model

from ts_vad.tasks.ts_vad import TSVADTaskConfig
from ts_vad.models.modules.modules import Conv1D, ChannelWiseLayerNorm, Conv1DBlock_v2, ConvTrans1D, ResBlock, Conv1DBlock
from ts_vad.models.modules.losses import SISNRLoss

logger = logging.getLogger(__name__)


NON_LINEAR = ChoiceEnum(["relu", "sigmoid", "softmax"])

@dataclass
class SpexPlusConfig(FairseqDataclass):
    non_linear: NON_LINEAR = field(
        default="relu",
        metadata={"help": "path to pretrained speaker encoder path."}
    )
    enc_in_channels: int = field(
        default=256,
        metadata={"help": "encoder in channel size"}
    )
    enc_out_channels: int = field(
        default=20,
        metadata={"help": "encoder out channel size"}
    )
    enc_out_size: int = field(
        default=256,
        metadata={"help": "encoder out size"}
    )
    conv_out_channels: int = field(
        default=512,
        metadata={"help": "conv out channels"}
    )
    conv_kernel_size: int = field(
        default=3,
        metadata={"help": "conv kernel size"}
    )
    conv_block_num: int = field(
        default=8,
        metadata={"help": "conv block num"}
    )
    total_conv_block_num: int = field(
        default=4,
        metadata={"help": "total conv block num"}
    )
    conv_block_norm: str = field(
        default="cLN",
        metadata={"help": "conv block norm"}
    )
    sisnr_eps: float = field(
        default=1.0e-06,
        metadata={"help": "si_snr eps"},
    )

@register_model("spex_plus", dataclass=SpexPlusConfig)
class SpexPlusModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: SpexPlusConfig,
        task_cfg: TSVADTaskConfig,
        spk_embed,
    ) -> None:
        super().__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if cfg.non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(cfg.non_linear))
        self.non_linear_type = cfg.non_linear
        self.non_linear = supported_nonlinear[cfg.non_linear]

        # Multi-scale Encoder
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.L1 = cfg.enc_out_channels
        self.L2 = 80
        self.L3 = 160
        self.encoder_1d_short = Conv1D(1, cfg.enc_in_channels, cfg.enc_out_channels, stride=cfg.enc_out_channels // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, cfg.enc_in_channels, 80, stride=cfg.enc_out_channels // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, cfg.enc_in_channels, 160, stride=cfg.enc_out_channels // 2, padding=0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(3 * cfg.enc_in_channels)
        # n x N x T => n x B x T
        self.proj = Conv1D(3 * cfg.enc_in_channels, cfg.enc_out_size, 1)

        # Repeat Conv Blocks 
        # n x B x T => n x B x T
        self.conv_blocks = nn.ModuleList([
                Conv1DBlock_v2(
                    spk_embed_dim=256, 
                    in_channels=cfg.enc_out_size, 
                    conv_channels=cfg.conv_out_channels, 
                    kernel_size=cfg.conv_kernel_size, 
                    norm=cfg.conv_block_norm, 
                    dilation=1
                ) for _ in range(cfg.total_conv_block_num)
            ]
        )
        self.conv_blocks_other = nn.ModuleList([
                self._build_blocks(
                    num_blocks=cfg.conv_block_num, 
                    in_channels=cfg.enc_out_size, 
                    conv_channels=cfg.conv_out_channels, 
                    kernel_size=cfg.conv_kernel_size, 
                    norm=cfg.conv_block_norm
                ) for _ in range(cfg.total_conv_block_num)
            ]
        )

        # Multi-scale Decoder
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = th.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask1 = Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1)
        self.mask2 = Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1)
        self.mask3 = Conv1D(cfg.enc_out_size, cfg.enc_in_channels, 1)

        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_1 = ConvTrans1D(cfg.enc_in_channels, 1, kernel_size=cfg.enc_out_channels, stride=cfg.enc_out_channels // 2, bias=True)
        self.decoder_1d_2 = ConvTrans1D(cfg.enc_in_channels, 1, kernel_size=80, stride=cfg.enc_out_channels // 2, bias=True)
        self.decoder_1d_3 = ConvTrans1D(cfg.enc_in_channels, 1, kernel_size=160, stride=cfg.enc_out_channels // 2, bias=True)

        # Speaker Encoder
        self.aux_enc3 = nn.Sequential(
            ChannelWiseLayerNorm(3 * 256),
            Conv1D(3 * 256, 256, 1),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            Conv1D(512, 256, 1),
        )
        self.pred_linear = nn.Linear(256, len(spk_embed))
        self.num_updates = 0

        self.se_loss = SISNRLoss(cfg.sisnr_eps)
        self.spk_loss = torch.nn.CrossEntropyLoss()
        self.inference = task_cfg.inference
        if self.inference:
            self.rs_len = task_cfg.rs_len
            self.segment_shift = task_cfg.segment_shift

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(1, num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def create_segments(self, mix_speech, mix_speech_len, tgt_speech):
        _, L = mix_speech.size()
        for start in range(0, mix_len - rs_len, dis):
            pass

    def forward(
        self,
        mix_speech,
        mix_speech_len,
        tgt_speech,
        aux_speech,
        aux_speech_len,
        spk_id,
    ):
        if self.inference:
            assert mix_speech.size(0) == 1

        if mix_speech.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, mix_speech.dim()))
        # when inference, only one utt
        if mix_speech.dim() == 1:
            mix_speech = torch.unsqueeze(mix_speech, 0)

        # Multi-scale Encoder (Mixture audio input)
        w1 = F.relu(self.encoder_1d_short(mix_speech))
        T = w1.shape[-1]
        xlen1 = mix_speech.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(mix_speech, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(mix_speech, (0, xlen3 - xlen1), "constant", 0)))
        # n x 3N x T
        y = self.ln(torch.cat([w1, w2, w3], 1))
        # n x B x T
        y = self.proj(y)

        # Multi-scale Encoder (Reference audio input)
        aux_w1 = F.relu(self.encoder_1d_short(aux_speech))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux_speech.shape[-1]
        aux_len2 = (aux_T_shape - 1) * (self.L1 // 2) + self.L2
        aux_len3 = (aux_T_shape - 1) * (self.L1 // 2) + self.L3
        aux_w2 = F.relu(self.encoder_1d_middle(F.pad(aux_speech, (0, aux_len2 - aux_len1), "constant", 0)))
        aux_w3 = F.relu(self.encoder_1d_long(F.pad(aux_speech, (0, aux_len3 - aux_len1), "constant", 0)))

        # Speaker Encoder
        aux = self.aux_enc3(torch.cat([aux_w1, aux_w2, aux_w3], 1))        
        aux_T = (aux_speech_len - self.L1) // (self.L1 // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = torch.sum(aux, -1)/aux_T.view(-1,1).float()

        # Speaker Extractor
        for conv_block, conv_block_other in zip(self.conv_blocks, self.conv_blocks_other):
            y = conv_block(y, aux)
            y = conv_block_other(y)

        # Multi-scale Decoder
        m1 = self.non_linear(self.mask1(y))
        m2 = self.non_linear(self.mask2(y))
        m3 = self.non_linear(self.mask3(y))
        s1 = w1 * m1
        s2 = w2 * m2
        s3 = w3 * m3

        ests = self.decoder_1d_1(s1, squeeze=True)
        ests2 = self.decoder_1d_2(s2, squeeze=True)[:, :xlen1]
        ests3 = self.decoder_1d_3(s3, squeeze=True)[:, :xlen1]
        spk_pred = self.pred_linear(aux)

        loss_se_1 = 0.0
        loss_se_2 = 0.0
        loss_se_3 = 0.0
        if tgt_speech.size(1) != ests.size(1):
            ests = torch.nn.functional.pad(ests, (0, tgt_speech.size(1) - ests.size(1), 0, 0))
        if tgt_speech.size(1) != ests2.size(1):
            ests2 = torch.nn.functional.pad(ests2, (0, tgt_speech.size(1) - ests2.size(1), 0, 0))
        if tgt_speech.size(1) != ests.size(1):
            ests3 = torch.nn.functional.pad(ests3, (0, tgt_speech.size(1) - ests3.size(1), 0, 0))
        for i in range(ests.size(0)):
            loss_se_1 += self.se_loss(tgt_speech[i, :mix_speech_len[i]].view(1, -1), ests[i, :mix_speech_len[i]].view(1, -1))
            loss_se_2 += self.se_loss(tgt_speech[i, :mix_speech_len[i]].view(1, -1), ests2[i, :mix_speech_len[i]].view(1, -1))
            loss_se_3 += self.se_loss(tgt_speech[i, :mix_speech_len[i]].view(1, -1), ests3[i, :mix_speech_len[i]].view(1, -1))

        loss_se_1 = loss_se_1 / ests.size(0)
        loss_se_2 = loss_se_2 / ests.size(0)
        loss_se_3 = loss_se_3 / ests.size(0)
        loss_se = 0.8 * loss_se_1 + 0.1 * loss_se_2 + 0.1 * loss_se_3

        result = {
            "losses": {
                "se_loss": loss_se
            },
            "se_1": loss_se_1.item(),
            "se_2": loss_se_2.item(),
            "se_3": loss_se_3.item(),
        }

        if self.inference:
            result["wave"] = ests
            return None, result
        else:
            spk_loss = self.spk_loss(spk_pred, spk_id)
            spk_acc = self.spk_accuracy(spk_pred, spk_id)[0]

            result['losses']['spk_loss'] = spk_loss
            result['spk_acc'] = spk_acc
            return result

    def spk_accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        # target is the correct speaker id
        output = output.detach()
        target = target.detach()

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @classmethod
    def build_model(cls, cfg: SpexPlusConfig, task: TSVADTaskConfig):
        """Build a new model instance."""

        model = SpexPlusModel(cfg, task.cfg, task.speaker_dictionary)
        return model
