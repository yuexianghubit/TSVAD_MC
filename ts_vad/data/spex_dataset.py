from collections import defaultdict
import logging
import os
from scipy import signal
import glob, json, random, wave

import numpy as np

import torch
import torchaudio
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from ts_vad.data.rttm import RttmReader

logger = logging.getLogger(__name__)


class SpexDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        rttm_path: str,
        rs_len: int,
        segment_shift: int = 6,
        musan_path: str = None,
        rir_path: str = None,
        noise_ratio: float = 0.5,
        sample_rate: int = 8000,
        spk_dict = None,
        inference: bool = False,
    ):
        self.manifest_path = manifest_path
        self.rttm = RttmReader(rttm_path)
        rs_len = sample_rate * rs_len # Number of frames for reference speech

        self.data_list = []
        self.label_dic = defaultdict(list)

        lines = open(manifest_path).read().splitlines()
        self.sizes = []
        # Load the data and labels
        for line in lines:
            utt_id, mix_path, tgt_path, aux_path, mix_len, spk_id = line.strip('\n').split('\t')
            mix_len = int(mix_len)

            if inference:
                self.data_list.append([utt_id, mix_path, tgt_path, aux_path, spk_id, 0, mix_len])
                self.sizes.append(mix_len)
            else:
                dis = sample_rate * segment_shift
                for start in range(0, mix_len, dis):
                    end = (start + rs_len) if start + rs_len < mix_len else mix_len
                    if end - start > sample_rate * 1.0:
                        data_intro = [utt_id, mix_path, tgt_path, aux_path, spk_id, start, end]
                        self.data_list.append(data_intro)
                        self.sizes.append(end - start)
                    
                    if end == mix_len:
                        break

        self.musan_path = musan_path
        if musan_path is not None:
            self.noiselist = {}
            self.noisetypes = ['noise', 'speech', 'music']
            self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
            self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
            augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            for file in augment_files:
                if file.split('/')[-3] not in self.noiselist:
                    self.noiselist[file.split('/')[-3]] = []
                self.noiselist[file.split('/')[-3]].append(file)
        self.rir_path = rir_path
        if rir_path is not None:
            self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

        self.noise_ratio = noise_ratio
        self.spk_dict = spk_dict
        self.sample_rate = sample_rate

        logger.info(
            f"loaded sentence={len(self.sizes)} "
            f"rs_len={rs_len}, segment_shift={segment_shift}, rir={rir_path is not None}, "
            f"musan={musan_path is not None}, noise_ratio={noise_ratio}, "
            f"shortest sent={min(self.sizes)}, longest sent={max(self.sizes)}"
        )

    def __getitem__(self, index):
        # T: number of frames (1 frame = 0.04s)
        # ref_speech : 16000 * (T / 25)
        # labels : 4, T
        # target_speech: 4, 16000 * (T / 25)
        utt_id, mix_path, tgt_path, aux_path, spk_id, start, end = self.data_list[index]
        mix_speech = self.load_audio(mix_path, start, end, add_noise=True)
        tgt_speech = self.load_audio(tgt_path, start, end)
        aux_speech = self.load_audio(aux_path)

        sample = {
            'id': index,
            'utt_id': utt_id,
            'mix_speech': mix_speech,
            'tgt_speech': tgt_speech,
            'aux_speech': aux_speech,
        }

        if spk_id in self.spk_dict:
            sample["spk_id"] = self.spk_dict(spk_id)
        else:
            sample["spk_id"] = -1

        return sample

    def load_audio(self, file, start=None, stop=None, add_noise=False):
        if start is not None:
            audio, cur_sample_rate = torchaudio.load(file, frame_offset=start, num_frames=stop - start)
            assert cur_sample_rate == self.sample_rate
            audio = audio.view(-1).numpy()
        else:
            audio = self.read_audio_with_resample(file)
        audio = np.expand_dims(np.array(audio), axis = 0)
        frame_len = audio.shape[1]

        if (self.rir_path is not None or self.musan_path is not None) and add_noise:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                if self.rir_path is not None and self.musan_path is not None:
                    noise_type = random.randint(0, 1)
                    if noise_type == 0:
                        audio = self.add_rev(audio, length=frame_len)
                    elif noise_type == 1:
                        audio = self.choose_and_add_noise(random.randint(0, 2), audio, frame_len)
                elif self.rir_path is not None:
                    audio = self.add_rev(audio, length=frame_len)
                elif self.musan_path is not None:
                    audio = self.choose_and_add_noise(random.randint(0, 2), audio, frame_len)

        audio = torch.FloatTensor(np.array(audio)).view(-1)

        return audio

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)

    def collater(self, samples):
        if len([s["spk_id"] for s in samples]) == 0:
            return {}
        mix_speech = _collate_frames([s["mix_speech"] for s in samples], is_audio_input=True)
        mix_speech_len = torch.tensor([s["mix_speech"].size(0) for s in samples], dtype=torch.long)

        tgt_speech = _collate_frames([s["tgt_speech"] for s in samples], is_audio_input=True)

        aux_speech = _collate_frames([s["aux_speech"] for s in samples], is_audio_input=True)
        aux_speech_len = torch.tensor([s["aux_speech"].size(0) for s in samples], dtype=torch.long)

        spk_id = torch.tensor([s["spk_id"] for s in samples], dtype=torch.long)

        net_input = {
            "mix_speech": mix_speech,
            "mix_speech_len": mix_speech_len,
            "tgt_speech": tgt_speech,
            "aux_speech": aux_speech,
            "aux_speech_len": aux_speech_len,
            "spk_id": spk_id,
        }

        batch = {
            "id": torch.LongTensor([s['id'] for s in samples]),
            "utt_id": [s['utt_id'] for s in samples],
            "net_input": net_input,
        }

        return batch

    def ordered_indices(self):
        order = [np.random.permutation(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def add_rev(self, audio, length):
        rir_file    = random.choice(self.rir_files)
        rir         = self.read_audio_with_resample(rir_file)
        rir         = np.expand_dims(rir.astype(np.float), 0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:length]

    def add_noise(self, audio, noisecat, length):
        clean_db    = 10 * np.log10(max(1e-4, np.mean(audio ** 2)))
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, 'rb').getnframes()
            noise_sample_rate = wave.open(noise, 'rb').getframerate()
            if noise_sample_rate != self.sample_rate:
                noiselength = int(noiselength * self.sample_rate / noise_sample_rate)
            if noiselength <= length:
                noiseaudio = self.read_audio_with_resample(noise)
                noiseaudio = np.pad(noiseaudio, (0, length - noiselength), 'wrap')
            else:
                start_frame = np.int64(random.random()*(noiselength-length))
                noiseaudio  = self.read_audio_with_resample(noise, start=start_frame, length=length)
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio ** 2)))
            noisesnr   = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise[:, :length] + audio

    def choose_and_add_noise(self, noise_type, ref_speech, frame_len):
        assert self.musan_path is not None
        if noise_type == 0:
            return self.add_noise(ref_speech, 'speech', length=frame_len)
        elif noise_type == 1:
            return self.add_noise(ref_speech, 'music', length=frame_len)
        elif noise_type == 2:
            return self.add_noise(ref_speech, 'noise', length=frame_len)

    def read_audio_with_resample(self, audio_path, start=None, length=None):
        cur_sample_rate = torchaudio.info(audio_path).sample_rate
        if start is not None:
            audio, cur_sample_rate = torchaudio.load(audio_path, frame_offset=int(start * (cur_sample_rate / self.sample_rate)), num_frames=int(length * (cur_sample_rate / self.sample_rate)))
        else:
            audio, cur_sample_rate = torchaudio.load(audio_path)
        if cur_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, cur_sample_rate, self.sample_rate).view(-1).numpy()

        return audio
