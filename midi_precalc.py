import os

import librosa
import numpy as np
import pretty_midi
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

PITCHES_COUNT = 128


def midi_processing(midi_path, hop_size, sr, audio_size, midi_dir):
    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    out = np.zeros((audio_size, PITCHES_COUNT), dtype=np.uint8)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            out[
                round((note.start * sr) / hop_size) : round((note.end * sr) / hop_size),
                note.pitch,
            ] = 1
    out_compress = csr_matrix(out)
    with open(
        f"{midi_dir}/{os.path.splitext(os.path.basename(midi_path))[0]}.npz", "wb"
    ) as f:
        save_npz(f, out_compress)


def process_audio(**kwargs):
    audio_file = kwargs["audio_file"]
    midi_file = kwargs["midi_file"]
    hop_size = kwargs["hop_size"]
    new_sr = kwargs.get("new_sr", None)
    waveform, sr = librosa.load(audio_file, sr=None)
    if sr != new_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=new_sr)
        sr = new_sr
    # waveform_cqt = librosa.cqt(
    #    waveform,
    #    sr=sr,
    #    hop_length=hop_size,
    #    n_bins=144,
    #    bins_per_octave=24,
    #    fmin=librosa.note_to_hz("C1"),
    # )
    # waveform_cqt = np.abs(
    #    waveform_cqt
    # )  # column_count = floor(len(waveform)/hop_size) + 1
    # if 1 + (len(waveform) // hop_size) != waveform_cqt.shape[1]:
    #        print("oops")
    _ = midi_processing(
        midi_file,
        hop_size,
        sr,
        1 + (len(waveform) // hop_size),
        midi_dir="midi_dir/train",
    )


audio_dir = "./dataset/train"
midi_dir = "./dataset/train"
new_sr = 44100
hop_size = 1024
audio_files = [f for f in os.listdir(audio_dir) if f.endswith((".wav", ".mp3"))]
midi_files = [f for f in os.listdir(midi_dir) if f.endswith((".midi"))]
midi_path_list = []
assert len(audio_files) == len(
    midi_files
), f"Audio files count: {len(audio_files)} but MIDI files count: {len(midi_files)}"

audio_files.sort()
midi_files.sort()

for audio_file, midi_file in tqdm(zip(audio_files, midi_files)):
    name_audio = os.path.splitext(audio_file)[0]
    name_midi = os.path.splitext(midi_file)[0]
    if name_audio != name_midi:
        raise ValueError(f"Mismatch: {name_audio}.midi and {name_midi}.* audio file")
    audio_path = os.path.join(audio_dir, audio_file)
    midi_path = os.path.join(midi_dir, midi_file)
    params = {
        "audio_file": audio_path,
        "midi_file": midi_path,
        "hop_size": hop_size,
        "new_sr": new_sr,
        "transform": None,
    }
    process_audio(**params)
