import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
from data.utils import draw


def pad_and_mask_batch(batch):
    strokes, sentences = zip(*batch)
    stroke_lengths = [len(x) for x in strokes]
    sentence_lengths = [len(x) for x in sentences]
    bsz = len(batch)

    stroke_arr = torch.zeros(bsz, max(stroke_lengths), 3).float()
    stroke_mask = torch.zeros(bsz, max(stroke_lengths)).float()

    sent_arr = torch.zeros(bsz, max(sentence_lengths)).long()
    sent_mask = torch.zeros(bsz, max(sentence_lengths)).float()

    for i, (stroke, length) in enumerate(zip(strokes, stroke_lengths)):
        stroke_arr[i, :length] = stroke
        stroke_mask[i, :length] = 1.

    for i, (sent, length) in enumerate(zip(sentences, sentence_lengths)):
        sent_arr[i, :length] = sent
        sent_mask[i, :length] = 1.

    return stroke_arr, stroke_mask, sent_arr, sent_mask


class HandwritingDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        root = Path(path)
        self.strokes = np.load(root / 'strokes.npy', encoding='latin1')
        self.sentences = open(root / 'sentences.txt').read().splitlines()
        self.sentences = [list(x) for x in self.sentences]

        ctr = Counter()
        for line in self.sentences:
            ctr.update(line)

        self.vocab = sorted(list(ctr.keys()))
        self.char2idx = {x: i for i, x in enumerate(self.vocab)}
        self.max_stroke_len = 1200

    def __len__(self):
        return self.strokes.shape[0]

    def sent2idx(self, sent):
        return np.asarray([self.char2idx[c] for c in sent])

    def idx2sent(self, sent):
        return ''.join(self.vocab[i] for i in sent)

    def __getitem__(self, idx):
        stroke = self.strokes[idx][:self.max_stroke_len]
        stroke = torch.from_numpy(stroke).clamp(-50, 50)
        stroke[:, 1:] /= 10

        sentence = torch.from_numpy(
            self.sent2idx(self.sentences[idx])
        ).long()
        return stroke, sentence


if __name__ == '__main__':
    path = '/Tmp/kumarrit/iam_ondb'
    dataset = HandwritingDataset('./lyrebird_data')
    loader = DataLoader(dataset, batch_size=16, collate_fn=pad_and_mask_batch)
    for i, (stk, stk_mask, sent, sent_mask) in tqdm(enumerate(loader)):
        if i == 0:
            print(stk.shape)
            print(stk_mask.shape)
            print(sent.shape)
            print(sent_mask.shape)

            for i in range(16):
                print(dataset.idx2sent(sent[i].tolist()))
                draw(stk[i].numpy(), save_file='test.png')
                input()