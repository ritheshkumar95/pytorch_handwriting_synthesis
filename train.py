from utils import draw
from dataset import HandwritingDataset, pad_and_mask_batch
from modules import Seq2Seq
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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

    with torch.no_grad():
        itr = iter(sampling_loader)
        for i in range(8):
            data = itr.__next__()
            data = [x.cuda() for x in data]
            (chars, chars_mask, strokes, strokes_mask) = data
            stroke_loss, eos_loss, att, _, sample = model(
                strokes, strokes_mask, chars, chars_mask
            )
            out = model.sample(chars, chars_mask).cpu().numpy()
            sample = sample.cpu().numpy()


            fig = plot_image(att['phi'].squeeze().cpu().numpy().T)
            writer.add_figure('attention/phi_%d' % i, fig, steps)

            fig = plot_lines(att['alpha'].squeeze().cpu().numpy().T)
            writer.add_figure('attention/alpha_%d' % i, fig, steps)

            fig = plot_lines(att['beta'].squeeze().cpu().numpy().T)
            writer.add_figure('attention/beta_%d' % i, fig, steps)

            fig = plot_lines(att['kappa'].squeeze().cpu().numpy().T)
            writer.add_figure('attention/kappa_%d' % i, fig, steps)

            fig = draw(
                out[0],
                save_file=root / ("generated_%d.png" % i)
            )
            writer.add_figure("samples/generated_%d" % i, fig, steps)

            fig = draw(
                sample[0],
                save_file=root / ("teacher_forced_%d.png" % i)
            )
            writer.add_figure("samples/teacher_forced_%d" % i, fig, steps)

            np.save(root / ('tf_%d.npy' % i), sample[0])


def train(epoch):
    global steps
    costs = []
    start_time = time.time()
    for iterno, data in enumerate(train_loader):
        data = [x.cuda() for x in data]
        (chars, chars_mask, strokes, strokes_mask) = data
        stroke_loss, eos_loss, att, prev_states, _ = model(
            strokes, strokes_mask,
            chars, chars_mask
        )

        opt.zero_grad()
        (stroke_loss + eos_loss).backward()
        for name, p in model.named_parameters():
            if 'lstm' in name:
                p.grad.data.clamp_(-10, 10)
            elif 'fc' in name:
                p.grad.data.clamp_(-100, 100)
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
            print("Took %5.4fs to generate samples" % (time.time() - st))
            print("-" * 100)


def test(epoch):
    global steps
    costs = []
    start_time = time.time()

    with torch.no_grad():
        for iterno, data in enumerate(test_loader):
            data = [x.cuda() for x in data]
            (chars, chars_mask, strokes, strokes_mask) = data

            stroke_loss, eos_loss, att, _, _ = model(
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

    parser.add_argument("--dec_hidden_size", type=int, default=400)
    parser.add_argument("--dec_n_layers", type=int, default=3)
    parser.add_argument("--n_mixtures_attention", type=int, default=10)
    parser.add_argument("--n_mixtures_output", type=int, default=20)
    parser.add_argument("--mask_loss", action='store_true')

    parser.add_argument("--path", default='./lyrebird_data')
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--save_interval", type=int, default=250)
    args = parser.parse_args()
    return args


args = parse_args()

root = Path(args.save_path)
load_root = Path(args.load_path) if args.load_path else None
if not root.exists():
    root.mkdir()
pickle.dump(args, open(root / "args.pkl", "wb"))
writer = SummaryWriter(str(root))

train_dataset = HandwritingDataset(args.path, split='train')
test_dataset = HandwritingDataset(args.path, split='test')
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    collate_fn=pad_and_mask_batch
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    collate_fn=pad_and_mask_batch
)
sampling_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=pad_and_mask_batch
)

model = Seq2Seq(
    train_dataset.vocab_size,
    args.dec_hidden_size, args.dec_n_layers,
    args.n_mixtures_attention, args.n_mixtures_output
).cuda()
print(model)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

if load_root and load_root.exists():
    model.load_state_dict(torch.load(load_root / 'model.pt'))

#######################################################################
# Dumping original data
#######################################################################
itr = iter(sampling_loader)
for i in range(8):
    data = itr.__next__()
    fig = draw(
        data[2][0].numpy(),
        save_file=root / ("original_%d.png" % i)
    )
    writer.add_figure("samples/original_%d" % i, fig, 0)

costs = []
start = time.time()
steps = 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    torch.save(model.state_dict(), root / "model.pt")
