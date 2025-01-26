import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import register_task, FairseqTask
import torch

#from ts_vad.data.spex_dataset import SpexDataset
from ts_vad.data.ts_vad_dataset import TSVADDataset
from ts_vad.data.spk_dictionary import SpkDictionary


logger = logging.getLogger(__name__)

SPEECH_ENCODER_TYPE = ChoiceEnum(["wavlm", "ecapa", "fbank"])
TASK_TYPE = ChoiceEnum(["diarization", "extraction"])

class LabelEncoder(object):
    def __init__(self, dictionary: SpkDictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str):
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )

    def __contains__(self, sym):
        return sym in self.dictionary

@dataclass
class TSVADTaskConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={"help": "path to data directory."}
    )
    mic_idx: int = field(
        default=-1,
        metadata={"help": "the index of microphone. if mic_idx=-1, return all microphones."}
    )
    ts_len: int = field(
        default=6000,
        metadata={"help": "Input ms of target speaker utterance"}
    )
    rs_len: int = field(
        default=6000,
        metadata={"help": "Input ms of reference speech"}
    )
    segment_shift: int = field(
        default=6,
        metadata={"help": "Speech shift during segmenting"}
    )
    spk_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to audio directory."}
    )
    musan_path: Optional[str] = field(
        default=None,
        metadata={"help": "musan path."}
    )
    rir_path: Optional[str] = field(
        default=None,
        metadata={"help": "rir path."}
    )
    speech_encoder_type: SPEECH_ENCODER_TYPE = field(
        default="ecapa",
        metadata={"help": "path to pretrained speaker encoder path."}
    )
    noise_ratio: float = field(
        default=0.5,
        metadata={"help": "noise ratio when adding noise"}
    )
    zero_ratio: float = field(
        default=0.5,
        metadata={"help": "the ratio to pad zero vector when shuffle level is 0"}
    )
    shuffle_spk_embed_level: int = field(
        default=0,
        metadata={"help": "shuffle spk embedding"}
    )
    task_type: TASK_TYPE = field(
        default="diarization",
        metadata={"help": "task type"}
    )
    sample_rate: int = field(
        default=8000,
        metadata={"help": "sample rate for input audio of SE task"}
    )

    # Inf
    min_silence: float = field(
        default=0.32,
        metadata={"help": "min silence"}
    )
    min_speech: float = field(
        default=0.0,
        metadata={"help": "min speech"}
    )
    inference: bool = field(
        default=False,
        metadata={"help": "inference or not"}
    )

@register_task("ts_vad_task", dataclass=TSVADTaskConfig)
class TSVADTask(FairseqTask):
    """
    This task is responsible for code input tasks.
    If pre-training, then code is the input. No explicit output is provided.
    If fine-tuning, then code is the input, and ltr is the output.
    """
    def __init__(self, cfg: TSVADTaskConfig):
        super().__init__(cfg)
        self.cfg = cfg
        if cfg.task_type == "extraction":
            assert os.path.isfile(f"{self.cfg.data}/dict.spk.txt")
            self.state.add_factory("speaker_dictionary", self.load_dictionary)

    def load_dictionary(self):
        return SpkDictionary.load(f"{self.cfg.data}/dict.spk.txt")

    @property
    def speaker_dictionary(self) -> Optional[SpkDictionary]:
        return self.state.speaker_dictionary

    def load_dataset(
        self,
        split: str,
        **kwargs,
    ):
        if self.cfg.task_type == "diarization":
            spk_path = self.cfg.spk_path
            # if 'train' in split.lower() and self.cfg.shuffle_spk_embed_level != 0:
            #     spk_path = f"{self.cfg.spk_path}/{split}/ecapa_feature_dir"
            # else:
            #     if os.path.isdir(f"{self.cfg.spk_path}/{split}/ecapa_avg_feature_dir"):
            #         spk_path = f"{self.cfg.spk_path}/{split}/ecapa_avg_feature_dir"
            #     else:
            #         spk_path = f"{self.cfg.spk_path}/{split}/ecapa_feature_dir"

            self.datasets[split] = TSVADDataset(
                json_path=f"{self.cfg.data}/ts_infer.json",
                audio_path=f"{self.cfg.data}/target_audio",
                mic_idx=self.cfg.mic_idx,
                ts_len=self.cfg.ts_len,
                rs_len=self.cfg.rs_len,
                spk_path=spk_path,
                is_train='train' in split.lower(),
                segment_shift=self.cfg.segment_shift,
                musan_path=self.cfg.musan_path if 'train' in split.lower() else None,
                rir_path=self.cfg.rir_path if 'train' in split.lower() else None,
                noise_ratio=self.cfg.noise_ratio,
                shuffle_spk_embed_level=self.cfg.shuffle_spk_embed_level,
                zero_ratio=self.cfg.zero_ratio,
            )
        elif self.cfg.task_type == "extraction":
            self.datasets[split] = SpexDataset(
                manifest_path=f"{self.cfg.data}/{split}.tsv",
                rttm_path=f"{self.cfg.data}/{split}.rttm",
                rs_len=self.cfg.rs_len,
                segment_shift=self.cfg.segment_shift,
                musan_path=self.cfg.musan_path if 'train' in split.lower() else None,
                rir_path=self.cfg.rir_path if 'train' in split.lower() else None,
                noise_ratio=self.cfg.noise_ratio,
                sample_rate=self.cfg.sample_rate,
                spk_dict=LabelEncoder(self.speaker_dictionary),
                inference=self.cfg.inference,
            )

    def inference_step(
        self, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return models[0](**sample['net_input'])

    @classmethod
    def setup_task(
        cls, cfg: TSVADTaskConfig, **kwargs
    ) -> "TSVADTaskConfig":
        return cls(cfg)
