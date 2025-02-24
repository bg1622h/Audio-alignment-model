import torch


def padding(notes,zero_vec):
    max_size = 0
    for batch in notes:
        max_size = len(batch)
        print(batch.shape)
    for i in range(len(notes)):
        while (len(notes[i]) < max_size):
            notes[i].append(zero_vec)
    notes = torch.tensor(notes,dtype=torch.int32)
    print(notes.shape)
    return notes

def collate_fn(batch):
    audio = [item['audio'] for item in batch]
    notes = [item['notes'] for item in batch]
    audio = torch.stack(audio)
    notes = torch.stack(notes)
    #notes = torch.nn.utils.rnn.pad_sequence(notes, batch_first=True, padding_value=0)
    return {
        'audio': audio,
        'notes': notes,
    }