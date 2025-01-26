from collections import defaultdict
import logging
import os
from scipy import signal
import glob, json, random, wave

import numpy as np
import soundfile as sf

import torch
import torchaudio.compliance.kaldi as kaldi
from fairseq.data.fairseq_dataset import FairseqDataset
from ts_vad.utils.make_to_onehot import make_to_onehot_by_scatter

logger = logging.getLogger(__name__)


class TSVADDataset(FairseqDataset):
    def __init__(
        self,
        json_path: str,
        audio_path: str,
        mic_idx: int,
        ts_len: int,
        rs_len: int,
        is_train: bool,
        spk_path: str = None,
        segment_shift: int = 6,
        musan_path: str = None,
        rir_path: str = None,
        noise_ratio: float = 0.5,
        shuffle_spk_embed_level: int = 0,
        zero_ratio: float = 0.5,
    ):
        self.audio_path = audio_path
        self.spk_path = spk_path
        self.mic_idx = mic_idx
        self.ts_len = int(ts_len / 40) # Number of frames for target speech
        self.rs_len = int(rs_len / 40) # Number of frames for reference speech

        self.data_list = []
        self.label_dic = defaultdict(list)
        self.label_piw_dic = defaultdict(list)

        lines = open(json_path).read().splitlines()
        filename_set = set()
        self.sizes = []
        self.spk2data = {}
        self.data2spk = {}
        # Load the data and labels
        for line in lines:
            dict = json.loads(line)
            length = len(dict['labels']) # Number of frames (1s = 25 frames)
            filename = dict['filename']
            labels = dict['labels']
            labels_piw = dict['labels_piw']
            speaker_id = str(dict['speaker_key'])
            speaker_id_full = str(dict['speaker_id'])

            if speaker_id_full not in self.spk2data:
                self.spk2data[speaker_id_full] = []
            self.spk2data[speaker_id_full].append(filename + '/' + speaker_id)
            self.data2spk[filename + '/' + speaker_id] = speaker_id_full

            full_id = filename + '_' + speaker_id
            self.label_dic[full_id] = labels
            self.label_piw_dic[full_id] = labels_piw
            if filename in filename_set:
                pass
            else:
                filename_set.add(filename)
                dis = 25 * segment_shift
                for start in range(0, length - self.rs_len, dis):
                    folder = self.audio_path + '/' + filename + '/*.wav'
                    audios = glob.glob(folder)
                    num_speaker = len(audios) - 1 # The total number of speakers, 2 or 3 or 4
                    data_intro = [filename, num_speaker, start, start + self.rs_len]
                    self.data_list.append(data_intro)
                    self.sizes.append(self.rs_len)

        self.musan_path = musan_path
        if musan_path is not None:
            self.noiselist = {}
            self.noisetypes = ['noise', 'speech', 'music']
            self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
            self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
            augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
            for file in augment_files:
                if file.split('/')[-3] not in self.noiselist:
                    self.noiselist[file.split('/')[-3]] = []
                self.noiselist[file.split('/')[-3]].append(file)
        
        self.rir_path = rir_path
        if rir_path is not None:
            self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

        self.noise_ratio = noise_ratio
        self.zero_ratio = zero_ratio
        self.shuffle_spk_embed_level = shuffle_spk_embed_level
        if self.shuffle_spk_embed_level != 0:
            assert self.spk_path is not None

        self.is_train = is_train

        logger.info(
            f"shuffle_spk_embed={shuffle_spk_embed_level}, "
            f"rs_len={rs_len}, segment_shift={segment_shift}, rir={rir_path is not None}, "
            f"musan={musan_path is not None}, noise_ratio={noise_ratio}"
        )

    def __getitem__(self, index):
        # T: number of frames (1 frame = 0.04s)
        # ref_speech : 16000 * (T / 25)
        # labels : 4, T
        # target_speech: 4, 16000 * (T / 25)
        file, num_speaker, start, stop = self.data_list[index]
        speaker_ids = self.get_ids(num_speaker)
        ref_speech, labels, labels_piw, new_speaker_ids = self.load_rs(file, speaker_ids, start, stop, self.mic_idx)
        if self.spk_path is None:
            target_speech = self.load_ts(file, speaker_ids)
        else:
            target_speech = self.load_ts_embed(file, new_speaker_ids)

        sample = {
            'id': index, 
            'ref_speech': ref_speech,
            'target_speech': target_speech, 
            'labels': labels,
            'labels_piw': labels_piw,
            'file_path': file,
            'speaker_ids': np.array(speaker_ids),
            'start': np.array(start),
        }

        return sample

    def get_ids(self, num_speaker):
        speaker_ids = []
        if num_speaker == 2:
            if self.shuffle_spk_embed_level not in [2, 3]:
                speaker_ids = [1, 2, 1, 2]
            else:
                speaker_ids = [1, 2, -1, -1]
        elif num_speaker == 3:
            if self.shuffle_spk_embed_level not in [2, 3]:
                speaker_ids = [1, 2, 3, random.randint(1, 3)]
            else:
                speaker_ids = [1, 2, 3, -1]
        else:
            speaker_ids = [1, 2, 3, 4]
        if self.is_train:
            random.shuffle(speaker_ids)
        return speaker_ids

    def load_rs(self, file, speaker_ids, start, stop, mic_idx):
        ref_speech, _ = sf.read(os.path.join(self.audio_path, file + '/all.wav'), start = start * 640, stop = stop * 640 + 240) # Since 25 * 640 = 16000

        # mic_idx > 0, it is the single-channel setting
        if mic_idx > 0: 
            ref_speech = ref_speech[:, mic_dix]

        # This part is data argumentation stategy
        ref_speech = np.expand_dims(np.array(ref_speech), axis = 0) # (1, T, C)
        frame_len = int(self.rs_len / 25 * 16000) + 240

        if self.rir_path is not None or self.musan_path is not None:
            add_noise = np.random.choice(2, p=[1 - self.noise_ratio, self.noise_ratio])
            if add_noise == 1:
                if self.rir_path is not None and self.musan_path is not None:
                    noise_type = random.randint(0, 1)
                    if noise_type == 0:
                        ref_speech = self.add_rev(ref_speech, length=frame_len)
                    elif noise_type == 1:
                        ref_speech = self.choose_and_add_noise(random.randint(0, 2), ref_speech, frame_len)
                elif self.rir_path is not None:
                    ref_speech = self.add_rev(ref_speech, length=frame_len)
                elif self.musan_path is not None:
                    ref_speech = self.choose_and_add_noise(random.randint(0, 2), ref_speech, frame_len)

        ref_speech = ref_speech[0] # (T, C)
        ref_speech = torch.FloatTensor(np.array(ref_speech))

        labels = []
        labels_piw = []
        new_speaker_ids = []
        for speaker_id in speaker_ids:
            if speaker_id == -1:
                labels.append(np.zeros(stop - start)) # Obatin the labels for the reference speech
                labels_piw.append(np.zeros(stop - start))
            else:
                full_label_id = file + '_' + str(speaker_id)
                label = self.label_dic[full_label_id] # sometimes label = []
                labels.append(label[start:stop]) # Obatin the labels for the reference speech
                label_piw = self.label_piw_dic[full_label_id] 
                labels_piw.append(label_piw[start:stop])
            if sum(labels[-1]) == 0 and self.shuffle_spk_embed_level in [2, 3] and self.is_train:
                new_speaker_ids.append(-1)
            else:
                new_speaker_ids.append(speaker_id)

        labels = torch.from_numpy(np.array(labels)).float() # 4, T
        labels_piw = torch.from_numpy(np.array(labels_piw)).long() # 4, T

        return ref_speech, labels, labels_piw, new_speaker_ids
 
    def load_ts_embed(self, file, speaker_ids):
        target_speeches = []
        if self.shuffle_spk_embed_level in [2, 3]:
            exist_spk = []
            for speaker_id in speaker_ids:
                if speaker_id != -1:
                    exist_spk.append(self.data2spk[f"{file}/{speaker_id}"])
        for speaker_id in speaker_ids:
            if speaker_id == -1:
                if self.shuffle_spk_embed_level == 3 and (np.random.choice(2, p=[1 - self.zero_ratio, self.zero_ratio]) == 1 or not self.is_train):
                    feature = torch.zeros(192)
                else:
                    random_spk = random.choice(list(self.spk2data))
                    while random_spk in exist_spk:
                        random_spk = random.choice(list(self.spk2data))
                    exist_spk.append(random_spk)
                    path = os.path.join(self.spk_path, f"{random.choice(self.spk2data[random_spk])}.pt")
                    feature = torch.load(path, map_location='cpu')
            else:
                path = os.path.join(self.spk_path, file, str(speaker_id) + '.pt')
                feature = torch.load(path, map_location='cpu')
            if len(feature.size()) == 2:
                if self.shuffle_spk_embed_level != 0 and self.is_train:
                    feature = feature[random.randint(0, feature.shape[0] - 1),:]
                else:
                    feature = torch.mean(feature, dim = 0)
            target_speeches.append(feature)
        target_speeches = torch.stack(target_speeches)
        return target_speeches

    def load_ts(self, file, speaker_ids):
        target_speeches = []
        for speaker_id in speaker_ids:
            path = os.path.join(self.audio_path, file, str(speaker_id) + '.wav')
            wav_length = wave.open(path, 'rb').getnframes() # entire length for target speech
            start = np.int64(random.random()*(wav_length-int(self.ts_len / 25 * 16000) - 240)) # start point
            frame_len = int(self.ts_len / 25 * 16000) + 240
            stop = start + frame_len
            target_speech, _ = sf.read(path, start=start, stop=stop)

            target_speech = np.expand_dims(np.array(target_speech), axis = 0)
            target_speech = target_speech[0]

            target_speech = torch.FloatTensor(np.array(target_speech))
            target_speech = (target_speech * (1 << 15)).unsqueeze(0)			
            target_speech = kaldi.fbank(target_speech, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0, sample_frequency=16000, window_type='hamming', use_energy=False)
            target_speech = torch.permute(target_speech, (1, 0))

            target_speeches.append(target_speech)
        target_speeches = torch.stack(target_speeches) # 4, 16000 * (T / 25)
        return target_speeches

    def __len__(self):
        return len(self.sizes)

    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.size(index)

    def collater(self, samples):
        ref_speech = torch.stack([s["ref_speech"] for s in samples], dim=0)
        target_speech = torch.stack([s["target_speech"] for s in samples], dim=0)
        labels = torch.stack([s["labels"] for s in samples], dim=0)
        labels_piw = torch.stack([s["labels_piw"] for s in samples], dim=0)

        net_input = {
            "ref_speech": ref_speech, 
            "target_speech": target_speech, 
            "labels": labels,
            "labels_piw": labels_piw,
            "file_path": [s["file_path"] for s in samples],
            "speaker_ids": [s["speaker_ids"] for s in samples],
            "start": [s["start"] for s in samples],
        }

        batch = {
            "id": torch.LongTensor([s['id'] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def ordered_indices(self):
        order = [np.random.permutation(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def add_rev(self, audio, length):
        # TODO: the RIRs in multi-chnanel audio is different, so we need to change.
        rir_file    = random.choice(self.rir_files)
        rir, sr     = sf.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float), 0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        reverb_signal = np.zeros_like(audio)
        for idx in range(audio.shape[-1]):
            reverb_signal[:,:,idx] = signal.convolve(audio[:,:,idx], rir, mode='full')[:,:length]
        return reverb_signal

    def add_noise(self, audio, noisecat, length):
        clean_db    = 10 * np.log10(max(1e-4, np.mean(audio ** 2)))
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, 'rb').getnframes()
            if noiselength <= length:
                noiseaudio, _ = sf.read(noise)
                noiseaudio = np.pad(noiseaudio, (0, length - noiselength), 'wrap')
            else:
                start_frame = np.int64(random.random()*(noiselength-length))
                noiseaudio, _ = sf.read(noise, start=start_frame, stop=start_frame + length)
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(max(1e-4, np.mean(noiseaudio ** 2)))
            noisesnr   = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True)
        noisy_audio = np.zeros_like(audio)
        for idx in range(audio.shape[-1]):
            noisy_audio[:,:,idx] = noise + audio[:,:,idx]
        return noisy_audio

    def choose_and_add_noise(self, noise_type, ref_speech, frame_len):
        assert self.musan_path is not None
        if noise_type == 0:
            return self.add_noise(ref_speech, 'speech', length=frame_len)
        elif noise_type == 1:
            return self.add_noise(ref_speech, 'music', length=frame_len)
        elif noise_type == 2:
            return self.add_noise(ref_speech, 'noise', length=frame_len)
