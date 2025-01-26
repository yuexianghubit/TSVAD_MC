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

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from ts_vad.tasks.ts_vad import TSVADTaskConfig
from ts_vad.models.modules.speakerEncoder import ECAPA_TDNN, PreEmphasis
from ts_vad.models.modules.postional_encoding import PositionalEncoding
from ts_vad.models.modules.WavLM import WavLM, WavLMConfig
from ts_vad.models.modules.joint_speaker_det import JointSpeakerDet
from ts_vad.models.modules.batch_norm import BatchNorm1D
from ts_vad.models.modules.self_att import CoAttention_Simple, CoAttention
from ts_vad.utils.make_to_onehot import make_to_onehot_by_scatter

logger = logging.getLogger(__name__)


@dataclass
class TSVADConfig(FairseqDataclass):
    speaker_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pretrained speaker encoder path."}
    )
    speech_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pretrained speech encoder path."}
    )
    freeze_speech_encoder_updates: int = field(
        default=10000,
        metadata={"help": "updates to freeze speech encoder."}
    )

    num_attention_head: int = field(
        default=8,
        metadata={"help": "number of attention head."}
    )
    num_transformer_layer: int = field(
        default=3,
        metadata={"help": "number of transformer layer."}
    )
    transformer_embed_dim: int = field(
        default=384,
        metadata={"help": "transformer dimension."}
    )
    transformer_ffn_embed_dim: int = field(
        default=768,
        metadata={"help": "transformer dimension."}
    )
    speaker_embed_dim: int = field(
        default=192,
        metadata={"help": "speaker embedding dimension."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout prob"}
    )
    use_jsd_block: bool = field(
        default=False,
        metadata={"help": "number of JSD block."}
    )
    cut_silence: bool = field(
        default=False,
        metadata={"help": "cut silence during train and inference."}
    )

@register_model("ts_vad", dataclass=TSVADConfig)
class TSVADModel_SC(BaseFairseqModel):
    def __init__(
        self,
        cfg: TSVADConfig,
        task_cfg: TSVADTaskConfig,
    ) -> None:
        super().__init__()
        # Speaker Encoder
        if task_cfg.spk_path is None:
            self.speaker_encoder = ECAPA_TDNN(C=1024)
            self.speaker_encoder.train()
            self.load_ecapa(cfg.speaker_encoder_path)
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
        else:
            self.use_spk_embed = True
        self.rs_dropout = nn.Dropout(p=cfg.dropout)

        # Speech Encoder
        self.speech_encoder_type = task_cfg.speech_encoder_type
        if self.speech_encoder_type == "wavlm":
            checkpoint = torch.load(cfg.speech_encoder_path, map_location="cuda")
            wavlm_cfg  = WavLMConfig(checkpoint['cfg'])
            wavlm_cfg.encoder_layers = 6
            self.speech_encoder = WavLM(wavlm_cfg)
            self.speech_encoder.train()
            self.speech_encoder.load_state_dict(checkpoint['model'], strict = False)
            self.speech_down = nn.Sequential(
                nn.Conv1d(768, cfg.speaker_embed_dim, 5, stride=2, padding=2),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "ecapa":
            self.speech_encoder = ECAPA_TDNN(C=1024, dropout=cfg.dropout)
            self.speech_encoder.train()
            self.load_ecapa(cfg.speech_encoder_path, module_name='speech_encoder')

            self.speech_down = nn.Sequential(
                nn.Conv1d(1536, cfg.speaker_embed_dim, 5, stride=4, padding=2),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )
        elif self.speech_encoder_type == "fbank":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),            
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                    f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
            self.speech_up = nn.Sequential(
                nn.Conv1d(80, cfg.speaker_embed_dim, 5, stride=4, padding=2),
                BatchNorm1D(num_features=cfg.speaker_embed_dim),
                nn.ReLU(),
            )

        # CoAttention
        #self.transformer_attn = SelfAttention(out_channels=192, embed_dim=256, num_heads=256)
        self.co_attn = CoAttention_Simple(out_channels=192, embed_dim=256, num_heads=256) #修改这里来做单通道
        #self.co_attn = CoAttention(out_channels=192, embed_dim=256, num_heads=256)

        # Projection
        if cfg.speaker_embed_dim * 2 != cfg.transformer_embed_dim:
            self.proj_layer = nn.Linear(cfg.speaker_embed_dim * 2, cfg.transformer_embed_dim)
        else:
            self.proj_layer = None

        # TS-VAD Backend
        self.single_backend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.transformer_embed_dim, 
                dim_feedforward=cfg.transformer_ffn_embed_dim, 
                nhead=cfg.num_attention_head, 
                dropout=cfg.dropout,
            ), 
            num_layers=cfg.num_transformer_layer
        )

        self.pos_encoder = PositionalEncoding(cfg.transformer_embed_dim, dropout=cfg.dropout, max_len=(task_cfg.rs_len // 40))
        self.use_jsd_block = cfg.use_jsd_block
        if self.use_jsd_block:
            self.multi_backend = JointSpeakerDet(cfg)

            # final projection
            self.fc = nn.Linear(cfg.transformer_embed_dim, 1)
            #self.fc_piw = nn.Linear(cfg.transformer_embed_dim, 5) # 5 classes
        else:
            self.backend_down = nn.Sequential(
                nn.Conv1d(cfg.transformer_embed_dim * 4, cfg.transformer_embed_dim, 5, stride=1, padding=2),
                BatchNorm1D(num_features=cfg.transformer_embed_dim),
                nn.ReLU(),
            )
            self.multi_backend = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.transformer_embed_dim, 
                    dim_feedforward=cfg.transformer_ffn_embed_dim, 
                    nhead=cfg.num_attention_head, 
                    dropout=cfg.dropout,
                ),
                num_layers=cfg.num_transformer_layer
            )
            # final projection
            self.fc = nn.Linear(cfg.transformer_embed_dim // 4, 1)
            #self.fc_piw = nn.Linear(cfg.transformer_embed_dim // 4, 7) # 7 classes

        if cfg.cut_silence:
            self.loss = nn.BCEWithLogitsLoss(reduction = 'none')
        else:
            self.loss = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

        if cfg.cut_silence:
            self.loss_piw = nn.BCEWithLogitsLoss(reduction = 'none')
        else:
            self.loss_piw = nn.BCEWithLogitsLoss()

        # others
        self.freeze_speech_encoder_updates = cfg.freeze_speech_encoder_updates
        self.cut_silence = cfg.cut_silence
        self.inference = task_cfg.inference
        self.num_updates = 0

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    ## B: batchsize, T: number of frames (1 frame = 0.04s)
    ## Obtain the reference speech represnetation
    #def rs_forward(self, x, fix_encoder=False): # B, 25 * T
    #    B, T = x.size()
    #    if self.speech_encoder_type == "wavlm":
    #        with torch.no_grad() if fix_encoder else contextlib.ExitStack():
    #            x = self.speech_encoder.extract_features(x)[0]        
    #        x = x.view(B, -1, 768)  # B, 50 * T, 768
    #        x = x.transpose(1, 2)
    #        x = self.speech_down(x)
    #        x = x.transpose(1, 2) # B, 25 * T, 256
    #    elif self.speech_encoder_type == "ecapa":
    #        with torch.no_grad() if fix_encoder else contextlib.ExitStack():
    #            x = self.speech_encoder(x, get_time_out=True)
    #        x = self.speech_down(x)
    #        x = x[:,:,:(T // 16000) * 25]
    #        x = x.transpose(1, 2)
    #    elif self.speech_encoder_type == "fbank":
    #        with torch.no_grad():
    #            with torch.cuda.amp.autocast(enabled=False): 
    #                x = self.torchfbank(x)+1e-6
    #                x = x.log()   
    #                x = x - torch.mean(x, dim=-1, keepdim=True)
    #        x = self.speech_up(x)
    #        x = x[:,:,:(T // 16000) * 25]
    #        x = x.transpose(1, 2)
    #    return x

    # B: batchsize, T: number of frames (1 frame = 0.04s), C: number of microphones
    # Obtain the reference speech represnetation
    def rs_forward(self, x, fix_encoder=False): # B, 25 * T
        B, T, C = x.size()
        x = x.transpose(1, 2) # B, C, T
        x = x.reshape(B * C, T)

        if self.speech_encoder_type == "wavlm":
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder.extract_features(x)[0]        
            x = x.view(B, -1, 768)  # B, 50 * T, 768
            x = x.transpose(1, 2)
            x = self.speech_down(x)
            x = x.transpose(1, 2) # B, 25 * T, 256
        elif self.speech_encoder_type == "ecapa":
            with torch.no_grad() if fix_encoder else contextlib.ExitStack():
                x = self.speech_encoder(x, get_time_out=True)
            x = self.speech_down(x)
            x = x[:,:,:(T // 16000) * 25]
            x = x.transpose(1, 2)
        elif self.speech_encoder_type == "fbank":
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False): 
                    x = self.torchfbank(x)+1e-6
                    x = x.log()   
                    x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.speech_up(x)
            x = x[:,:,:(T // 16000) * 25]
            x = x.transpose(1, 2)

        # Co-Attention
        x = x.reshape(B, C, x.size()[-2], x.size()[-1]) # B, C, T, D
        x = self.co_attn(x) # B, C, T, D --> B, T, D

        return x

    # Obtain the target speaker represnetation
    def ts_forward(self, x): # B, 4, 80, T * 100 
        if self.use_spk_embed:
            return self.rs_dropout(x)
        B, _, D, T = x.shape
        x = x.view(B*4, D, T)
        x = self.speaker_encoder.forward(x)
        x = x.view(B, 4, -1) # B, 4, 192
        return x

    # Combine for ts-vad results
    def cat_forward(self, rs_embeds, ts_embeds):
        # Extend ts_embeds for time alignemnt
        ts_embeds = ts_embeds.unsqueeze(2) # B, 4, 1, 256
        ts_embeds = ts_embeds.repeat(1, 1, rs_embeds.shape[1], 1) # B, 4, T, 256
        B, _, T, _ = ts_embeds.shape

        # # Extend rs_embeds for speaker alignemnt
        # rs_embeds = rs_embeds.unsqueeze(1) # B, 1, T, 256
        # rs_embeds = rs_embeds.repeat(1, 4, 1, 1) # B, 4, T, 256
        # # Transformer for single speaker
        # cat_embeds = torch.cat((ts_embeds, rs_embeds), -1) # B, T, 256 + B, T, 256 -> B, T, 512
        # cat_embeds = cat_embeds.view(B * 4, T, cat_embeds.size(-1)) # B * 4, T, 256
        # if self.proj_layer is not None:
        #     cat_embeds = self.proj_layer(cat_embeds)
        # cat_embeds = self.pos_encoder(cat_embeds.transpose(0, 1)).transpose(0, 1)
        # cat_embeds = self.single_backend(cat_embeds)

        # Transformer for single speaker
        cat_embeds = []
        for i in range(4):
            ts_embed = ts_embeds[:, i, :, :] # B, T, 256
            cat_embed = torch.cat((ts_embed, rs_embeds), 2) # B, T, 256 + B, T, 256 -> B, T, 512
            cat_embed = cat_embed.transpose(0, 1) # T, B, 512
            if self.proj_layer is not None:
                cat_embeds = self.proj_layer(cat_embeds)
            cat_embed = self.pos_encoder(cat_embed)
            cat_embed = self.single_backend(cat_embed) # T, B, 512
            cat_embed = cat_embed.transpose(0, 1) # B, T, 512
            cat_embeds.append(cat_embed)
        cat_embeds = torch.stack(cat_embeds) # 4, B, T, 384
        cat_embeds = torch.permute(cat_embeds, (1, 0, 3, 2)) # B, 4, 384, T
        # Combine the outputs
        cat_embeds = cat_embeds.reshape(B, -1, T)  # B, 4 * 384, T
        # Downsampling
        if self.use_jsd_block:
            cat_embeds = cat_embeds.reshape(B, 4, T, -1) # B, 4 * 512, T
            cat_embeds = self.multi_backend(cat_embeds) # B, S, T, D
        else:
            cat_embeds = self.backend_down(cat_embeds)  # B, 384, T
            # Transformer for multiple speakers
            cat_embeds = self.pos_encoder(torch.permute(cat_embeds, (2, 0, 1)))
            cat_embeds = self.multi_backend(cat_embeds).transpose(0, 1) # B, T, 384
            # Results for each speaker
            cat_embeds = cat_embeds.reshape((B, 4, T, -1))  # B, 4, T, 96

        # cat_embeds = torch.stack(cat_embeds) # 4, B, T, 512
        # # Transformer for multiple speakers on T-axis
        # cat_embeds = cat_embeds.reshape(T, 4 * B, -1) # T, 4 * B, 512
        # # cat_embeds = self.pos_encoder(cat_embeds)
        # cat_embeds = self.multi_backend_time(cat_embeds) # T, 4 * B, 512
        # # Transformer for multiple speakers on S-axis
        # cat_embeds = cat_embeds.reshape(4, T * B, -1) # 4, T * B, 512
        # cat_embeds = self.multi_backend_spk(cat_embeds) # 4, T * B, 512
        # # Results for each speaker
        # cat_embeds = cat_embeds.reshape((B, 4, T, -1))  # B, 4, T, 512
        return cat_embeds

    def calculate_loss(self, outs, labels):
        total_loss = 0
        if self.cut_silence:
            silence_labels = torch.sum(labels, dim = 1)
            silence_labels = torch.where(silence_labels >= 1, 1, 0)

        for i in range(4):
            output = outs[:, i, :]
            label = labels[:, i, :]
            loss = self.loss(output, label)

            if self.cut_silence:
                total_loss += torch.sum(silence_labels * loss) / torch.sum(silence_labels)
            else:
                total_loss += loss

        outs_prob = self.m(outs)
        if self.cut_silence:
            silence_labels = silence_labels.unsqueeze(1)
            silence_labels = silence_labels.repeat(1, 4, 1)
            outs_prob = outs_prob * silence_labels
        outs_prob = outs_prob.data.cpu().numpy()

        return total_loss / 4, outs_prob

    def calculate_mtl_loss(self, outs_1, labels_1, outs_2, labels_2):
        total_loss = 0
        if self.cut_silence:
            silence_labels = torch.sum(labels_1, dim = 1)
            silence_labels = torch.where(silence_labels_1 >= 1, 1, 0)

        for i in range(4):
            output_1 = outs_1[:, i, :]
            label_1 = labels_1[:, i, :]
            output_2 = outs_2[:, i, :]
            label_2 = nn.functional.one_hot(labels_2[:, i, :], 7).float()
            #print("label.size():", label_1.size())
            loss = self.loss(output_1, label_1) + 0.2 * self.loss_piw(output_2, label_2)

            if self.cut_silence:
                total_loss += torch.sum(silence_labels * loss) / torch.sum(silence_labels)
            else:
                total_loss += loss

        outs_prob = self.m(outs_1)
        if self.cut_silence:
            silence_labels = silence_labels.unsqueeze(1)
            silence_labels = silence_labels.repeat(1, 4, 1)
            outs_prob = outs_prob * silence_labels
        outs_prob = outs_prob.data.cpu().numpy()

        return total_loss / 4, outs_prob

    def forward(
        self,
        ref_speech: torch.Tensor,
        target_speech: torch.Tensor,
        labels: torch.Tensor,
        file_path,
        speaker_ids,
        start,
    ):
        rs_embeds  = self.rs_forward(ref_speech, fix_encoder=self.num_updates < self.freeze_speech_encoder_updates)  # rs_embeds(64,100,192)  ref_speech(64,64240,8)
        ts_embeds  = self.ts_forward(target_speech) #target_speech（64,4,192） ts_embeds(64,4,192)
        outs       = self.cat_forward(rs_embeds, ts_embeds)

        outs_sd = self.fc(outs).squeeze(-1)
        #outs_piw = self.fc_piw(outs).squeeze(-1)

        loss, outs_prob = self.calculate_loss(outs_sd, labels)
        #loss, outs_prob = self.calculate_mtl_loss(outs_sd, labels, outs_piw, labels_piw)
        result = {
            "losses":{
                "diar": loss
            }
        }

        # DER
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(outs_prob.transpose((0, 2, 1)), labels.transpose(1, 2))

        _, _, mi, fa, cf, acc, der = (
            speech_miss / speech_scored,
            speech_falarm / speech_scored,
            speaker_miss / speaker_scored,
            speaker_falarm / speaker_scored,
            speaker_error / speaker_scored,
            correct / num_frames,
            (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
        )

        result['DER'] = der
        result['ACC'] = acc
        result['MI'] = mi
        result['FA'] = fa
        result['CF'] = cf

        if self.inference:
            res_dict = defaultdict(lambda: defaultdict(list))
            B, _, T = outs_sd.shape
            for b in range(B):
                for t in range(T):
                    n = max(speaker_ids[b])
                    for i in range(n):
                        id = speaker_ids[b][i]
                        name = file_path[b]
                        out = outs_prob[b, i, t]
                        t0 = start[b]
                        res_dict[str(name) + '-' + str(id)][t0 + t].append(out)

            return result, res_dict

        return result

    @classmethod
    def build_model(cls, cfg: TSVADConfig, task: TSVADTaskConfig):
        """Build a new model instance."""

        model = TSVADModel(cfg, task.cfg)
        return model

    def load_ecapa(self, model_path, module_name="speaker_encoder"):
        loadedState = torch.load(model_path, map_location="cuda")
        selfState = self.state_dict()
        for name, param in loadedState.items():
            origname = name
            if isinstance(self.speech_encoder.bn1, BatchNorm1D) and '.'.join(name.split('.')[:-1]) + '.running_mean' in loadedState:
                name = '.'.join(name.split('.')[:-1]) + '.bn.' + name.split('.')[-1]

            if name.startswith('speaker_encoder'):
                name = name.replace('speaker_encoder', module_name)
            else:
                name = f'{module_name}.' + name

            if name not in selfState:
                print("%s is not in the model."%origname)
                continue
            if selfState[name].size() != loadedState[origname].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, selfState[name].size(), loadedState[origname].size()))
                continue
            selfState[name].copy_(param)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def calc_diarization_error(pred, label):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        # pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        pred_np = (pred > 0.5).astype(int)
        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np)) / num_output)
        num_frames = pred.shape[0] * pred.shape[1]
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )
