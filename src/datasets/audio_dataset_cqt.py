import os

import librosa
import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class MusicDataset_cqt_aug(Dataset):
    """
    Class, returns CQT for random fragment from audio (with specaug applied) of frame_size duration, along with its piano roll
    """

    def __init__(self, **kwargs):
        """
        Args:
            audio_file: path to audio file in .wav, .mp3 format
            midi_file - path to the corresponding record of compressed piano roll .npz
            hop_size,new_sr,n_bins,bins_per_octave,fmin - parameters for CQT
            frame_size - frame duration
            frame_counts - number of returned fragments
            min_pitch,max_pitch - Pianoroll only stores notes with min_pitch up to and max_pitch
        """
        self.audio_file = kwargs["audio_file"]
        self.midi_file = kwargs["midi_file"]
        self.hop_size = kwargs["hop_size"]
        self.new_sr = kwargs.get("new_sr", None)
        self.transform = kwargs.get("transform", None)
        self.frame_size = kwargs["frame_size"]
        self.min_pitch = kwargs.get("min_pitch", 24)
        self.max_pitch = kwargs.get("max_pitch", 96)
        self.n_bins = kwargs.get("n_bins", 216)
        self.bins_per_octave = kwargs.get("bins_per_octave", 36)
        self.fmin = kwargs.get("fmin", librosa.note_to_hz("C1"))
        self.frame_counts = kwargs.get("frame_counts", 2)

    def __len__(self):
        """
        Returns:
            The number of fragments to be returned
        """
        return self.frame_counts

    def __getitem__(self, index):
        """
        Args:
            index - not important
        Returns:
            dict{"audio": CQT for audio fragment with shape = (F,T), "notes": pianoroll for audio fragment with shape = (T,P)}
        """
        loaded_midi_data = load_npz(self.midi_file)
        midi_data = np.array(loaded_midi_data.toarray())
        midi_data = torch.tensor(midi_data, dtype=torch.float32)
        slice_length = (self.new_sr // self.hop_size + 1) * self.frame_size
        start_index = np.random.randint(0, midi_data.shape[0] - slice_length + 1)
        waveform, sr = librosa.load(
            self.audio_file,
            sr=None,
            duration=(slice_length * self.hop_size) / (self.new_sr),
            offset=(start_index * self.hop_size) / (self.new_sr),
        )
        if sr != self.new_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.new_sr)
            sr = self.new_sr
        waveform_cqt = librosa.cqt(
            waveform,
            sr=sr,
            hop_length=self.hop_size,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
        )
        waveform_cqt = np.abs(waveform_cqt)
        notes = midi_data[
            start_index : start_index + slice_length, self.min_pitch : self.max_pitch
        ]
        waveform_cqt = self.rfreq_masking(waveform_cqt)
        waveform_cqt = self.random_sparse_time_mask(waveform_cqt, notes)
        cqt_audio = torch.tensor(waveform_cqt, dtype=torch.float32)
        if slice_length > cqt_audio.shape[1]:
            assert slice_length == cqt_audio.shape[1]
        elif slice_length < cqt_audio.shape[1]:
            cqt_audio = cqt_audio[:, :slice_length]
        return {
            "audio": cqt_audio,
            "notes": notes,
        }

    def freq_masking(self, audio, max_mask_bins=12):
        """
        SpecAug, frequency masking
        Args:
            audio - CQT audio transform
        Returns:
            audio - CQT audio transform after frequency masking
        """
        mask_bins = np.random.randint(1, max_mask_bins)
        start = np.random.randint(0, audio.shape[0] - mask_bins)
        audio[start : start + mask_bins] = 0
        return audio

    def time_masking(self, audio, notes, max_mask_frames=30):
        """
        SpecAug, time masking
        Args:
            audio - CQT audio transform
            notes - pianoroll
        Returns:
            audio - CQT audio transform after time masking
            notes - pianoroll after time masking
        """
        mask_frames = np.random.randint(1, max_mask_frames)
        start = np.random.randint(0, audio.shape[1] - mask_frames)
        audio[:, start : start + mask_frames] = 0
        notes[start : start + mask_frames, :] = 0
        return audio, notes


class AudioDataset_cqt_aug(BaseDataset):
    """
    A class storing the MusicDataset_aug for a set of audio
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
        """
        Args:
            auido_dir - path to the folder with audio recordings of .wav, .mp3 format
            midi_dir - path to the corresponding records of compressed piano rolls .npz
            hop_size,new_sr - HCQT params
            frame_size - frame size
            dataset_size - Number of audio recordings used
        """
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
            params = {
                "audio_file": audio_path,
                "midi_file": midi_path,
                "hop_size": hop_size,
                "new_sr": new_sr,
                "transform": transform,
                "frame_size": frame_size,
            }
            datasets.append(MusicDataset_cqt_aug(**params))
        self.concat_datasets = ConcatDataset(datasets)

    def __len__(self):
        """
        Returns:
            The number of all fragments in all records
        """
        return len(self.concat_datasets)

    def __getitem__(self, index):
        """
        Returns:
            Random fragment
        """
        return self.concat_datasets[index]


class MusicDataset_cqt(Dataset):
    """
    Class storing CQT for all fragments of an audio recording of frame_size duration and the corresponding piano roll
    """

    def __init__(self, **kwargs):
        """
        Args:
            audio_file: path to audio file in .wav, .mp3 format
            midi_file - path to the corresponding record of compressed piano roll .npz
            hop_size,new_sr,n_bins,bins_per_octave,fmin - parameters for CQT
            frame_size - frame duration
            min_pitch, max_pitch - Pianoroll only stores notes with min_pitch up to and max_pitch
        """
        self.audio_file = kwargs["audio_file"]
        self.midi_file = kwargs["midi_file"]
        self.hop_size = kwargs["hop_size"]
        self.new_sr = kwargs.get("new_sr", None)
        self.transform = kwargs.get("transform", None)
        self.frame_size = kwargs["frame_size"]

        self.min_pitch = kwargs.get("min_pitch", 24)
        self.max_pitch = kwargs.get("max_pitch", 96)
        self.n_bins = kwargs.get("n_bins", 216)
        self.bins_per_octave = kwargs.get("bins_per_octave", 36)
        self.fmin = kwargs.get("fmin", librosa.note_to_hz("C1"))

        waveform, sr = librosa.load(self.audio_file, sr=None)
        if sr != self.new_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.new_sr)
            sr = self.new_sr
        waveform_cqt = librosa.cqt(
            waveform,
            sr=sr,
            hop_length=self.hop_size,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
        )
        waveform_cqt = np.abs(waveform_cqt)
        loaded_midi_data = load_npz(self.midi_file)
        midi_data = np.array(loaded_midi_data.toarray())
        audio = torch.tensor(waveform_cqt, dtype=torch.float32)
        midi_data = torch.tensor(midi_data, dtype=torch.float32)
        midi_data = midi_data[:, self.min_pitch : self.max_pitch]
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
                continue
            self.audio_segments.append(audio_segment)
            self.notes_segments.append(notes_segment)
        self.audio_segments = torch.stack(self.audio_segments, dim=0)
        self.notes_segments = torch.stack(self.notes_segments, dim=0)

    def __len__(self):
        """
        Returns:
            the number of fragments in the dataset
        """
        return len(self.audio_segments)

    def __getitem__(self, index):
        """
        Class returning an CQT transformation and the corresponding pianoroll fragment
        Args:
            index - fragment index
        Returns:
            dict{"audio": CQT for audio fragment with shape = (F,T), "notes": pianoroll for audio fragment with shape = (T,P)}
        """
        return {
            "audio": self.audio_segments[index],
            "notes": self.notes_segments[index],
        }


class AudioDataset_cqt(BaseDataset):
    """
    A class storing the MusicDataset_cqt for a set of audio
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
        """
        Args:
            auido_dir - path to the folder with audio recordings of .wav, .mp3 format
            midi_dir - path to the corresponding records of compressed piano rolls .npz
            hop_size,new_sr - CQT params
            frame_size - frame size
            is_train - If true then MusicDataset_hcqt will apply specaug, otherwise it will not
            dataset_size - Number of audio recordings used
        """
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
            params = {
                "audio_file": audio_path,
                "midi_file": midi_path,
                "hop_size": hop_size,
                "new_sr": new_sr,
                "transform": transform,
                "frame_size": frame_size,
            }
            datasets.append(MusicDataset_cqt(**params))

        self.concat_datasets = ConcatDataset(datasets)

    def __len__(self):
        """
        Returns:
            The number of all fragments in all records
        """
        return len(self.concat_datasets)

    def __getitem__(self, index):
        """
        Returns:
            Fragment with item index
        """
        return self.concat_datasets[index]
