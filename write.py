from utils import draw
from modules import HandwritingSynthesisNetwork
from dataset import HandwritingDataset, pad_and_mask_batch
from torch.utils.data import DataLoader

import pickle
import argparse
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", required=True)
    args = parser.parse_args()
    return args


new_args = parse_args()
root = Path(new_args.load_path)
args = pickle.load(open(root / "args.pkl", "rb"))

test_dataset = HandwritingDataset(args.path, split='test')
sampling_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=pad_and_mask_batch
)

model = HandwritingSynthesisNetwork(
    test_dataset.vocab_size,
    args.dec_hidden_size, args.dec_n_layers,
    args.n_mixtures_attention, args.n_mixtures_output
).cuda()
model.load_state_dict(torch.load(root / 'model.pt'))

while True:
    string = input("Enter input: ") + " "
    chars = torch.from_numpy(
        test_dataset.sent2idx(string)
    ).long()[None].cuda()
    chars_mask = torch.ones_like(chars).float().cuda()

    with torch.no_grad():
        out = model.sample(chars, chars_mask, maxlen=2000)[0].cpu().numpy()

    draw(out[0], save_file='./generated.jpg')
    print("Generated sample...\n")
