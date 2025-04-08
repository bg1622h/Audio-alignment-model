from pathlib import Path

import pandas as pd
import pretty_midi

"""
A small script to convert .csv note records in the SWD dataset into .midi files
"""


def get_csv_paths(directory):
    return list(Path(directory).rglob("*.csv"))


def create_midi(path):
    df = pd.read_csv(path, sep=";", skiprows=1, header=None)
    note_events = df.to_numpy()[:, :3]
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for event in note_events:
        note = pretty_midi.Note(
            velocity=100, pitch=int(event[2]), start=event[0], end=event[1]
        )
        instrument.notes.append(note)
    midi_data.instruments.append(instrument)
    new_path = path.with_suffix(".midi")
    midi_data.write(str(new_path))


directory = "./SWD"  # Path to directory with .csv notation
csv_files = get_csv_paths(directory)
for csv_file in csv_files:
    print(csv_file)
    create_midi(csv_file)
