from data.utils import draw, alphabet
from dataset import DataLoader
from modules import Seq2Seq
import matplotlib.pyplot as plt

import pickle
import numpy as np
import time
import argparse
import torch
from pathlib import Path
from tensorboardX import SummaryWriter


def plot_image(arr):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin='lower', aspect='auto', interpolation='nearest')
    fig.colorbar(im)
    return fig


def plot_lines(arr):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    for i in range(arr.shape[0]):
        ax.plot(arr[i], label='%d' % i)
    ax.legend()
    return fig


def plot_attention():
    global steps
    itr = loader.create_iterator('test', batch_size=1)

    with torch.no_grad():
        for i in range(8):
            chars, chars_mask, strokes, strokes_mask = itr.__next__()
            stroke_loss, eos_loss, att, _ = model(
                strokes, strokes_mask, chars, chars_mask
            )

            fig = plot_image(att.squeeze(0).cpu().numpy().T)
            writer.add_figure('attention/probs_%d' % i, fig, steps)


def train(epoch):
    global steps
    costs = []
    itr = loader.create_iterator('train', batch_size=args.batch_size)
    start_time = time.time()
    for iterno, (chars, chars_mask, strokes, strokes_mask) in enumerate(itr):
        stroke_loss, eos_loss, att, prev_states = model(
            strokes, strokes_mask,
            chars, chars_mask
        )

        opt.zero_grad()
        (stroke_loss + eos_loss).backward()
        for param in model.parameters():
            param.grad.clamp_(-10., 10.)
        opt.step()

        ####################################################
        costs.append([stroke_loss.item(), eos_loss.item()])

        writer.add_scalar("stroke_loss/train", costs[-1][0], steps)
        writer.add_scalar("eos_loss/train", costs[-1][1], steps)
        steps += 1

        if iterno % args.log_interval == 0:
            print(
                "Train Epoch {} Iterno {} | ms/batch {:5.2f} | loss {}".format(
                    epoch, iterno, 1000 * (time.time() - start_time) / len(costs),
                    np.asarray(costs).mean(0)
                )
            )
            start_time = time.time()
            costs = []

        if iterno % args.save_interval == 0:
            print("Saving samples....")
            st = time.time()
            plot_attention()
            with torch.no_grad():
                chars, chars_mask, _, _ = test_data
                out = model.sample(chars, chars_mask).detach().cpu().numpy()
                for i in range(8):
                    fig = draw(out[i], save_file=root / ("generated_%d.png" % i))
                    writer.add_figure("samples/generated_%d" % i, fig, steps)
            torch.cuda.empty_cache()
            print("Took %5.4fs to generate samples" % (time.time() - st))
            print("-" * 100)


def test(epoch):
    global steps
    costs = []
    itr = loader.create_iterator('test', batch_size=args.batch_size)
    start_time = time.time()

    with torch.no_grad():
        for iterno, (chars, chars_mask, strokes, strokes_mask) in enumerate(itr):
            stroke_loss, eos_loss, att, _ = model(
                strokes, strokes_mask, chars, chars_mask
            )
            costs.append([stroke_loss.item(), eos_loss.item()])

    stroke_loss, eos_loss = np.asarray(costs).mean(0)
    writer.add_scalar("stroke_loss/test", costs[-1][0], steps)
    writer.add_scalar("eos_loss/test", costs[-1][1], steps)

    print(
        "Test Epoch {} | ms/batch {:5.2f} | loss {}".format(
            epoch, 1000 * (time.time() - start_time) / len(costs),
            np.asarray(costs).mean(0)
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--enc_emb_size", type=int, default=256)
    parser.add_argument("--enc_hidden_size", type=int, default=400)
    parser.add_argument("--enc_n_layers", type=int, default=3)

    parser.add_argument("--dec_hidden_size", type=int, default=400)
    parser.add_argument("--dec_n_layers", type=int, default=3)
    parser.add_argument("--n_mixtures_output", type=int, default=20)
    parser.add_argument("--mask_loss", action='store_true')

    parser.add_argument("--path", default='./data/processed')
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--save_interval", type=int, default=250)
    args = parser.parse_args()
    return args


args = parse_args()
args.vocab_size = len(alphabet)
print('Vocab size: ', args.vocab_size)

root = Path(args.save_path)
load_root = Path(args.load_path) if args.load_path else None
if not root.exists():
    root.mkdir()
pickle.dump(args, open(root / "args.pkl", "wb"))
writer = SummaryWriter(str(root))

model = Seq2Seq(
    args.vocab_size, args.enc_emb_size,
    args.enc_hidden_size, args.enc_n_layers,
    args.dec_hidden_size, args.dec_n_layers,
    args.n_mixtures_output
).cuda()
print(model)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

if load_root and load_root.exists():
    model.load_state_dict(torch.load(load_root / 'model.pt'))

loader = DataLoader(args.path)
#######################################################################
# Dumping original data
#######################################################################
itr = loader.create_iterator('test', batch_size=8)
test_data = itr.__next__()
for i in range(8):
    fig = draw(test_data[2][i].cpu().numpy(), save_file=root / ("original_%d.png" % i))
    writer.add_figure("samples/original_%d" % i, fig, 0)

costs = []
start = time.time()
steps = 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    torch.save(model.state_dict(), root / "model.pt")
