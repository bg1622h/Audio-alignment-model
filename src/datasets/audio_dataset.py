import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
from scipy.sparse import load_npz
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class MusicDataset(Dataset):
    # One music audio
    # frame_size - in secs (maybe works with float too
    # audio_file, midi_file, hop_size, frame_size, transform=None, new_sr = None):
    def __init__(self, **kwargs):
        self.audio_file = kwargs["audio_file"]
        self.midi_file = kwargs["midi_file"]
        self.hop_size = kwargs["hop_size"]
        self.new_sr = kwargs.get("new_sr", None)
        self.transform = kwargs.get("transform", None)
        self.frame_size = kwargs["frame_size"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        loaded_midi_data = load_npz(self.midi_file)
        midi_data = np.array(loaded_midi_data.toarray())
        midi_data = torch.tensor(midi_data, dtype=torch.float32)
        slice_length = (self.new_sr // self.hop_size + 1) * self.frame_size
        start_index = np.random.randint(0, midi_data.shape[0] - slice_length + 1)
        # waveform, sr = librosa.load(self.audio_file,sr=None)
        waveform, sr = librosa.load(
            self.audio_file,
            sr=None,
            duration=(slice_length * self.hop_size) / (self.new_sr),
            offset=(start_index * self.hop_size) / (self.new_sr),
        )
        if sr != self.new_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.new_sr)
            sr = self.new_sr
        # waveform = waveform[
        #     start_index * self.hop_size : (start_index + slice_length) * self.hop_size
        # ]
        waveform_cqt = librosa.cqt(
            waveform,
            sr=sr,
            hop_length=self.hop_size,
            n_bins=144,
            bins_per_octave=24,
            fmin=librosa.note_to_hz("C1"),
        )
        waveform_cqt = np.abs(waveform_cqt)
        cqt_audio = torch.tensor(waveform_cqt, dtype=torch.float32)
        if slice_length > cqt_audio.shape[1]:
            print(slice_length)
            print(cqt_audio.shape[1])
            assert slice_length == cqt_audio.shape[1]
        elif slice_length < cqt_audio.shape[1]:
            cqt_audio = cqt_audio[:, :slice_length]
        return {
            "audio": cqt_audio,
            "notes": midi_data[start_index : start_index + slice_length, :],
        }


class AudioDataset(BaseDataset):
    """
    Dataset for audio and MIDI processing split into fixed 2-second segments.
    """

    def __init__(
        self,
        audio_dir,
        midi_dir,
        hop_size,
        frame_size,
        dataset_size=None,
        transform=None,
        new_sr=None,
    ):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.new_sr = new_sr
        self.hop_size = hop_size
        self.transform = transform
        self.frame_size = frame_size
        self.audio_files = [
            f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3"))
        ]
        self.midi_files = [f for f in os.listdir(midi_dir) if f.endswith((".npz"))]
        self.midi_path_list = []
        self.durations = []
        assert len(self.audio_files) == len(
            self.midi_files
        ), f"Audio files count: {len(self.audio_files)} but MIDI files count: {len(self.midi_files)}"

        self.audio_files.sort()
        self.midi_files.sort()

        if dataset_size is None:
            dataset_size = len(self.audio_files)

        self.audio_files = self.audio_files[:dataset_size]
        self.midi_files = self.midi_files[:dataset_size]

        datasets = []
        for audio_file, midi_file in zip(self.audio_files, self.midi_files):
            name_audio = os.path.splitext(audio_file)[0]
            name_midi = os.path.splitext(midi_file)[0]
            if name_audio != name_midi:
                raise ValueError(
                    f"Mismatch: {name_audio}.midi and {name_midi}.* audio file"
                )
            audio_path = os.path.join(self.audio_dir, audio_file)
            midi_path = os.path.join(self.midi_dir, midi_file)
            self.midi_path_list.append(midi_path)
            params = {
                "audio_file": audio_path,
                "midi_file": midi_path,
                "hop_size": hop_size,
                "new_sr": new_sr,
                "transform": transform,
                "frame_size": frame_size,
            }
            datasets.append(MusicDataset(**params))
            # self.durations.append(datasets[-1].get_duration())

        self.concat_datasets = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_datasets)

    def __getitem__(self, index):
        return self.concat_datasets[index]

    def get_midi_files(self, index):
        return self.midi_path_list[index]

    # def get_duration(self, index):
    #     return self.durations[index]


class test_MusicDataset(Dataset):
    # One music audio
    # frame_size - in secs (maybe works with float too)
    # audio_file, midi_file, hop_size, frame_size, transform=None, new_sr = None):
    def __init__(self, **kwargs):
        self.audio_file = kwargs["audio_file"]
        self.midi_file = kwargs["midi_file"]
        self.hop_size = kwargs["hop_size"]
        self.new_sr = kwargs.get("new_sr", None)
        self.transform = kwargs.get("transform", None)
        self.frame_size = kwargs["frame_size"]
        waveform, sr = librosa.load(self.audio_file, sr=None)
        self.duration = librosa.get_duration(y=waveform, sr=sr)
        if sr != self.new_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.new_sr)
            sr = self.new_sr
        waveform_cqt = librosa.cqt(
            waveform,
            sr=sr,
            hop_length=self.hop_size,
            n_bins=144,
            bins_per_octave=24,
            fmin=librosa.note_to_hz("C1"),
        )
        waveform_cqt = np.abs(
            waveform_cqt
        )  # column_count = floor(len(waveform)/hop_size) + 1
        loaded_midi_data = load_npz(self.midi_file)
        midi_data = np.array(loaded_midi_data.toarray())
        # midi_data = midi_processing(
        #    self.midi_file, self.hop_size, sr, waveform_cqt.shape[1]
        # )  # sec_per_column = (hop_length)/sr
        audio = torch.tensor(waveform_cqt, dtype=torch.float32)
        midi_data = torch.tensor(midi_data, dtype=torch.float32)
        segment_length = self.frame_size * (sr + self.hop_size - 1) // (self.hop_size)
        total_segments = (audio.size(1) + segment_length - 1) // segment_length
        self.audio_segments = []
        self.notes_segments = []
        for i in range(total_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            audio_segment = audio[:, start:end]
            notes_segment = midi_data[start:end, :]
            if audio_segment.size(1) < segment_length:
                audio_segment = torch.cat(
                    (
                        audio_segment,
                        torch.zeros(
                            audio_segment.size(0),
                            segment_length - audio_segment.size(1),
                        ),
                    ),
                    dim=1,
                )
                notes_segment = torch.cat(
                    (
                        notes_segment,
                        torch.zeros(
                            segment_length - notes_segment.size(0),
                            notes_segment.size(1),
                        ),
                    ),
                    dim=0,
                )
            self.audio_segments.append(audio_segment)
            self.notes_segments.append(notes_segment)
        self.audio_segments = torch.stack(self.audio_segments, dim=0)
        self.notes_segments = torch.stack(self.notes_segments, dim=0)

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, index):
        return {
            "audio": self.audio_segments[index],
            "notes": self.notes_segments[index],
        }

    def get_duration(self):
        return self.duration


class test_AudioDataset(BaseDataset):
    """
    Dataset for audio and MIDI processing split into fixed 2-second segments.
    """

    def __init__(
        self,
        audio_dir,
        midi_dir,
        hop_size,
        frame_size,
        dataset_size=None,
        transform=None,
        new_sr=None,
    ):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.new_sr = new_sr
        self.hop_size = hop_size
        self.transform = transform
        self.frame_size = frame_size
        self.audio_files = [
            f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3"))
        ]
        self.midi_files = [f for f in os.listdir(midi_dir) if f.endswith((".npz"))]
        self.midi_path_list = []
        self.durations = []
        assert len(self.audio_files) == len(
            self.midi_files
        ), f"Audio files count: {len(self.audio_files)} but MIDI files count: {len(self.midi_files)}"

        self.audio_files.sort()
        self.midi_files.sort()

        if dataset_size is None:
            dataset_size = len(self.audio_files)

        self.audio_files = self.audio_files[:dataset_size]
        self.midi_files = self.midi_files[:dataset_size]

        datasets = []
        for audio_file, midi_file in tqdm(zip(self.audio_files, self.midi_files)):
            name_audio = os.path.splitext(audio_file)[0]
            name_midi = os.path.splitext(midi_file)[0]
            if name_audio != name_midi:
                raise ValueError(
                    f"Mismatch: {name_audio}.midi and {name_midi}.* audio file"
                )
            audio_path = os.path.join(self.audio_dir, audio_file)
            midi_path = os.path.join(self.midi_dir, midi_file)
            self.midi_path_list.append(midi_path)
            params = {
                "audio_file": audio_path,
                "midi_file": midi_path,
                "hop_size": hop_size,
                "new_sr": new_sr,
                "transform": transform,
                "frame_size": frame_size,
            }
            datasets.append(test_MusicDataset(**params))
            self.durations.append(datasets[-1].get_duration())

        self.concat_datasets = ConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_datasets)

    def __getitem__(self, index):
        return self.concat_datasets[index]

    def get_midi_files(self, index):
        return self.midi_path_list[index]

    def get_duration(self, index):
        return self.durations[index]
