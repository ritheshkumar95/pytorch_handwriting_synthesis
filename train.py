from data.utils import draw
from dataset import HandwritingDataset, pad_and_mask_batch
from modules import RNNDecoder

import pickle
import numpy as np
import time
import argparse
import torch
from pathlib import Path
from tensorboardX import SummaryWriter


def train(epoch):
    global steps
    costs = []
    itr = loader.create_iterator('train', batch_size=args.batch_size, seq_len=args.seq_len)
    start_time = time.time()
    for iterno, (start, strokes, strokes_mask) in enumerate(itr):
        if start:
            prev_hidden = None
        else:
            prev_hidden = (prev_hidden[0].detach(), prev_hidden[1].detach())
        stroke_loss, eos_loss, prev_hidden = model.score(strokes, strokes_mask, prev_hidden)

        opt.zero_grad()
        (stroke_loss + eos_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        # for param in model.parameters():
        #     param.grad.clamp_(-10., 10.)
        opt.step()

        ####################################################
        costs.append([stroke_loss.item(), eos_loss.item()])

        writer.add_scalar("stroke_loss/train", costs[-1][0], steps)
        writer.add_scalar("eos_loss/train", costs[-1][1], steps)
        steps += 1

        if iterno % args.log_interval == 0:
            print(
                "Train Epoch {} Iterno {} | ms/batch {:5.2f} | loss {}".format(
                    epoch, iterno, 1000 * (time.time() - start_time) / args.log_interval,
                    np.asarray(costs).mean(0)
                )
            )
            start_time = time.time()
            costs = []

    print(
        "Train Epoch {} | ms/batch {:5.2f} | loss {}".format(
            epoch, 1000 * (time.time() - start) / len(costs),
            np.asarray(costs).mean(0)
        )
    )


def test(epoch):
    global steps
    costs = []
    itr = loader.create_iterator('test', batch_size=args.batch_size, seq_len=1200)
    start_time = time.time()
    for iterno, (start, strokes, strokes_mask) in enumerate(itr):
        if start:
            prev_hidden = None
        stroke_loss, eos_loss, prev_hidden = model.score(strokes, strokes_mask, prev_hidden)
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
    parser.add_argument("--n_mixtures_output", type=int, default=20)

    parser.add_argument("--path", default='/Tmp/kumarrit/iam_ondb')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=150)
    args = parser.parse_args()
    return args


args = parse_args()

root = Path(args.save_path)
load_root = Path(args.load_path) if args.load_path else None
if not root.exists():
    root.mkdir()
pickle.dump(args, open(root / "args.pkl", "wb"))
writer = SummaryWriter(str(root))

model = RNNDecoder(
    args.dec_hidden_size, args.dec_n_layers,
    args.n_mixtures_output
).cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

if load_root and load_root.exists():
    model.load_state_dict(torch.load(load_root / 'model.pt'))

loader = UnconditionalDataLoader(args.path)
#######################################################################
# Dumping original data
#######################################################################
itr = loader.create_iterator('test', batch_size=8)
test_data = itr.__next__()
for i in range(8):
    fig = draw(test_data[1][i].cpu().numpy(), save_file=root / ("original_%d.png" % i))
    writer.add_figure("samples/original_%d" % i, fig, 0)

costs = []
start = time.time()
steps = 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    print("Saving samples....")
    st = time.time()
    with torch.no_grad():
        out = model.sample(batch_size=8).detach().cpu().numpy()
        for i in range(8):
            fig = draw(out[i], save_file=root / ("generated_%d.png" % i))
            writer.add_figure("samples/generated_%d" % i, fig, steps)
    print("Took %5.4fs to generate samples" % (time.time() - st))
    print("-" * 100)

    torch.save(model.state_dict(), root / "model.pt")
