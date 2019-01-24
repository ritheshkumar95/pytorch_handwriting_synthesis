import numpy as np
import h5py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path


class UnconditionalDataLoader(object):
    def __init__(self, hdf5_path):    
        hdf5_path = Path(hdf5_path) / 'data.hdf5'
        self.h5 = h5py.File(hdf5_path, 'r')
        self.vocab = eval(self.h5.attrs['vocab'])
        self.char2idx = eval(self.h5.attrs['char2idx'])

        self.strokes = self.h5['strokes'][()]
        self.strokes_mask = self.h5['strokes_mask'][()]

        idxs = np.arange(self.strokes.shape[0])
        np.random.seed(111)
        np.random.shuffle(idxs)
        self.idxs = {
            'train' : idxs[:int(len(idxs) * .9)],
            'test' : idxs[int(len(idxs) * .9):]
        }

    def create_iterator(self, split='train', batch_size=64):
        idxs = self.idxs[split]
        for i in range(0, len(idxs), batch_size):
            stk = torch.from_numpy(self.strokes[idxs[i:i + batch_size]])[:, :600]
            stk_mask = torch.from_numpy(self.strokes_mask[idxs[i:i + batch_size]])[:, :600]

            # last = stk_mask.sum(0).argmin()
            # stk = stk[:, :last].cuda()
            # stk_mask = stk_mask[:, :last].cuda()

            yield stk.cuda(), stk_mask.cuda()


class HandwritingSynthDataset(torch.utils.data.Dataset):
    def __init__(
        self, hdf5_path
    ):
        hdf5_path = Path(hdf5_path) / 'data.hdf5'
        self.h5 = h5py.File(hdf5_path, 'r')
        self.vocab = eval(self.h5.attrs['vocab'])
        self.char2idx = eval(self.h5.attrs['char2idx'])

    def __getitem__(self, index):
        strokes = self.h5['strokes'][index][:300]
        strokes_mask = self.h5['strokes_mask'][index][:300]

        strokes = torch.from_numpy(strokes)
        strokes_mask = torch.from_numpy(strokes_mask)
        return strokes, strokes_mask

    def __len__(self):
        return len(self.h5['sentences'])


if __name__ == '__main__':
    path = '/tmp/kumarrit/iam_ondb'
    # dataset = HandwritingSynthDataset(path)
    # loader = DataLoader(dataset, batch_size=64, num_workers=0)
    # for i, data in tqdm(enumerate(loader)):
    #     if i == 0:
    #         for x in data:
    #             print(x.shape)

    loader = UnconditionalDataLoader(path)
    for split in ['train', 'test']:
        itr = loader.create_iterator(split)
        for i, data in tqdm(enumerate(itr)):
            if i == 0:
                for x in data:
                    print(x.shape)
