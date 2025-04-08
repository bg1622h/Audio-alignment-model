import os

import librosa
import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class MusicDataset_hcqt(Dataset):
    """
    Class storing HCQT for all fragments of an audio recording of frame_size duration (with specaug applied, if is_train = True) and the corresponding piano roll
    """

    def __init__(self, **kwargs):
        """
        Args:
            audio_file: path to audio file in .wav, .mp3 format
            midi_file - path to the corresponding record of compressed piano roll .npz
            hop_size,new_sr,n_bins,bins_per_octave,fmin - parameters for HCQT
            frame_size - frame duration
            min_pitch, max_pitch - Pianoroll only stores notes with min_pitch up to and max_pitch
            is_train - If true then MusicDataset_hcqt will apply specaug, otherwise it will not
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
        self.is_train = kwargs.get("is_train", False)

        harmonics = [0.5, 1, 2, 3, 4, 5]
        waveform, sr = librosa.load(self.audio_file, sr=None)
        if sr != self.new_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.new_sr)
            sr = self.new_sr
        tuning = librosa.estimate_tuning(y=waveform, sr=sr)
        hcqt = []
        target_frames = int(len(waveform) / self.hop_size)
        for h in harmonics:
            cqt = librosa.cqt(
                y=waveform,
                sr=sr,
                hop_length=self.hop_size,
                fmin=self.fmin * h,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                tuning=tuning,
            )
            cqt = librosa.util.fix_length(cqt, size=target_frames, axis=1)
            hcqt.append(np.abs(cqt))
        hcqt = np.stack(hcqt, axis=-1)
        loaded_midi_data = load_npz(self.midi_file)
        midi_data = np.array(loaded_midi_data.toarray())
        audio = torch.tensor(hcqt, dtype=torch.float32)
        audio = audio.permute(1, 0, 2)
        midi_data = torch.tensor(midi_data, dtype=torch.float32)
        midi_data = midi_data[:, self.min_pitch : self.max_pitch]
        segment_length = self.frame_size * (sr + self.hop_size - 1) // (self.hop_size)
        total_segments = (audio.size(0) + segment_length - 1) // segment_length
        self.audio_segments = []
        self.notes_segments = []
        for i in range(total_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            audio_segment = audio[start:end]
            notes_segment = midi_data[start:end, :]
            if (audio_segment.shape != (segment_length, self.n_bins, 6)) or (
                notes_segment.shape != (segment_length, self.max_pitch - self.min_pitch)
            ):
                continue
            self.audio_segments.append(audio_segment)
            self.notes_segments.append(notes_segment)
        self.audio_segments = [
            s
            for s in self.audio_segments
            if s.shape == (segment_length, self.n_bins, 6)
        ]
        self.notes_segments = [
            s
            for s in self.notes_segments
            if s.shape == (segment_length, self.max_pitch - self.min_pitch)
        ]
        try:
            self.audio_segments = torch.stack(self.audio_segments, dim=0)
            self.notes_segments = torch.stack(self.notes_segments, dim=0)
        except ValueError as e:
            print(e)
            self.audio_segments = []
            self.notes_segments = []

    def __len__(self):
        """
        Returns:
            the number of fragments in the dataset
        """
        return len(self.audio_segments)

    def time_masking(self, audio, notes, max_mask_frames=30):
        """
        SpecAug, time masking
        Args:
            audio - HCQT audio transform
            notes - pianoroll
        Returns:
            audio - HCQT audio transform after time masking
            notes - pianoroll after time masking
        """
        mask_frames = np.random.randint(1, max_mask_frames)
        start = np.random.randint(0, audio.shape[1] - mask_frames)
        audio[start : start + mask_frames] = 0
        notes[start : start + mask_frames, :] = 0
        return audio, notes

    def freq_masking(self, audio, max_mask_bins=12):
        """
        SpecAug, frequency masking
        Args:
            audio - HCQT audio transform
        Returns:
            audio - HCQT audio transform after frequency masking
        """
        mask_bins = np.random.randint(1, max_mask_bins)
        start = np.random.randint(0, audio.shape[0] - mask_bins)
        audio[:, start : start + mask_bins] = 0
        return audio

    def augmentations(self, audio, notes):
        """
        SpecAug augmentation
        Args:
            audio - HCQT audio transform
            notes - pianoroll
        Returns:
            audio,notes after SpecAug
        """
        if np.random.rand() > 0.5:
            audio, notes = self.time_masking(audio, notes)
        if np.random.rand() > 0.5:
            audio = self.freq_masking(audio)
        return audio, notes

    def __getitem__(self, index):
        """
        Class returning an HCQT transformation and the corresponding pianoroll fragment, optionally with SpecAug
        Args:
            index - fragment index
        Returns:
            dict{"audio": HCQT for audio fragment with shape = (T,F,6), "notes": pianoroll for audio fragment with shape = (T,P)}
        """
        if self.is_train:
            audio, notes = self.augmentations(
                self.audio_segments[index], self.notes_segments[index]
            )
        else:
            audio, notes = self.audio_segments[index], self.notes_segments[index]
        audio = audio.to(torch.float32)
        notes = notes.to(torch.float32)
        return {
            "audio": audio,
            "notes": notes,
        }


class AudioDataset_hcqt(BaseDataset):
    """
    A class storing the MusicDataset_hcqt for a set of audio
    """

    def __init__(
        self,
        audio_dir,
        midi_dir,
        hop_size,
        frame_size,
        is_train,
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
                "is_train": is_train,
            }
            datasets.append(MusicDataset_hcqt(**params))

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
