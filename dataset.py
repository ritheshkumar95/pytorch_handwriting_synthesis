import h5py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path


class HandwritingSynthDataset(torch.utils.data.Dataset):
    def __init__(
        self, hdf5_path
    ):
        hdf5_path = Path(hdf5_path) / 'data.hdf5'
        self.h5 = h5py.File(hdf5_path, 'r')
        self.vocab = eval(self.h5.attrs['vocab'])
        self.char2idx = eval(self.h5.attrs['char2idx'])

    def __getitem__(self, index):
        chars = self.h5['chars'][index]
        chars_mask = self.h5['chars_mask'][index]
        strokes = self.h5['strokes'][index]
        strokes_mask = self.h5['strokes_mask'][index]

        chars, chars_mask, strokes, strokes_mask = [
            torch.from_numpy(x)
            for x in [chars, chars_mask, strokes, strokes_mask]
        ]

        assert chars.min() >= 0
        assert chars.max() <= 82
        return chars, chars_mask, strokes, strokes_mask

    def __len__(self):
        return len(self.h5['sentences'])


if __name__ == '__main__':
    path = '/Tmp/kumarrit/iam_ondb'
    dataset = HandwritingSynthDataset(path)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    for i, data in tqdm(enumerate(loader)):
        if i == 0:
            for x in data:
                print(x.shape)
