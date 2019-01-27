import numpy as np
import h5py
import torch
from tqdm import tqdm
from pathlib import Path


class DataLoader(object):
    def __init__(self, hdf5_path):    
        hdf5_path = Path(hdf5_path) / 'data.hdf5'
        self.h5 = h5py.File(hdf5_path, 'r')
        self.vocab = eval(self.h5.attrs['vocab'])
        self.char2idx = eval(self.h5.attrs['char2idx'])

        self.chars = self.h5['chars'][()]
        self.chars_mask = self.h5['chars_mask'][()]
        self.strokes = self.h5['strokes'][()]
        self.strokes_mask = self.h5['strokes_mask'][()]
        stroke_lens = self.strokes_mask.sum(-1)

        idxs = list(zip(np.arange(len(stroke_lens)), stroke_lens))
        np.random.seed(111)
        np.random.shuffle(idxs)

        self.idxs = {}
        self.idxs['train'] = list(zip(*sorted(
            idxs[:int(len(idxs) * .9)], key=lambda tup: tup[1]
        )))
        self.idxs['test'] = list(zip(*sorted(
            idxs[int(len(idxs) * .9):], key=lambda tup: tup[1]
        )))

    def create_iterator(self, split='train', batch_size=64):
        idxs, stk_lengths = [np.array(x) for x in self.idxs[split]] 
        for i in range(0, len(idxs), batch_size):
            max_stk_len = int(max(stk_lengths[i:i + batch_size]))
            stk = torch.from_numpy(
                self.strokes[idxs[i:i + batch_size]][:, :max_stk_len]
            )
            stk_mask = torch.from_numpy(
                self.strokes_mask[idxs[i:i + batch_size]][:, :max_stk_len]
            )

            chars = torch.from_numpy(self.chars[idxs[i:i + batch_size]])
            chars_mask = torch.from_numpy(self.chars_mask[idxs[i:i + batch_size]])

            lengths, idx = torch.sort(chars_mask.sum(-1), 0, descending=True)
            stk = stk[idx].cuda()
            stk_mask = stk_mask[idx].cuda()
            chars = chars[idx].cuda()
            chars_mask = chars_mask[idx].cuda()

            yield chars, chars_mask, stk, stk_mask


if __name__ == '__main__':
    path = '/Tmp/kumarrit/iam_ondb'
    dataset = DataLoader(path)
    itr = dataset.create_iterator('train', batch_size=16)
    for i, data in tqdm(enumerate(itr)):
        if i == 0:
            for x in data:
                print(x.shape)
