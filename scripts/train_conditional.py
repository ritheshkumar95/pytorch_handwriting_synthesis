from utils import draw
from dataset import HandwritingDataset, pad_and_mask_batch
from modules import HandwritingSynthesisNetwork
from utils import plot_lines, plot_image
from torch.utils.data import DataLoader

import pickle
import numpy as np
import time
import argparse
import torch
from pathlib import Path
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--dec_hidden_size", type=int, default=400)
    parser.add_argument("--dec_n_layers", type=int, default=3)
    parser.add_argument("--n_mixtures_attention", type=int, default=10)
    parser.add_argument("--n_mixtures_output", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=600)

    parser.add_argument("--path", default='./lyrebird_data')
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()
    return args


def monitor_samples():
    itr = iter(sampling_loader)
    for i in range(8):
        data = itr.__next__()
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]

        with torch.no_grad():
            stroke_loss, eos_loss, monitor_vars, _, teacher_forced_sample = model.compute_loss(
                chars, chars_mask, strokes, strokes_mask
            )
            generated_sample = model.sample(chars, chars_mask)[0]

        teacher_forced_sample = teacher_forced_sample.cpu().numpy()
        generated_sample = generated_sample.cpu().numpy()

        # Plotting image for phi
        phi = monitor_vars.pop('phi')
        fig = plot_image(phi[0].squeeze().cpu().numpy().T)
        writer.add_figure('attention/phi_%d' % i, fig, steps)

        # Line plot for alpha, beta and kappa
        for key, val in monitor_vars.items():
            fig = plot_lines(val[0].cpu().numpy().T)
            writer.add_figure('attention/%s_%d' % (key, i), fig, steps)

        # Draw generated and teacher forced samples
        fig = draw(
            generated_sample[0], save_file=root / ("generated_%d.png" % i)
        )
        writer.add_figure("samples/generated_%d" % i, fig, steps)

        fig = draw(
            teacher_forced_sample[0], save_file=root / ("teacher_forced_%d.png" % i)
        )
        writer.add_figure("samples/teacher_forced_%d" % i, fig, steps)


def train(epoch):
    global steps
    costs = []
    start_time = time.time()
    for iterno, data in enumerate(train_loader):
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]
        seq_len = strokes.shape[1]

        prev_states = None
        for idx in range(1, seq_len, args.seq_len):
            stroke_loss, eos_loss, att, prev_states, _ = model.compute_loss(
                chars,
                chars_mask,
                strokes[:, idx - 1: idx + args.seq_len],
                strokes_mask[:, idx - 1: idx + args.seq_len],
                prev_states
            )

            prev_states = [
                (x[0].detach(), x[1].detach()) if type(x) is tuple else x.detach()
                for x in prev_states
            ]

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


def test(epoch):
    costs = []
    start_time = time.time()

    for iterno, data in enumerate(test_loader):
        chars, chars_mask, strokes, strokes_mask = [x.cuda() for x in data]

        with torch.no_grad():
            stroke_loss, eos_loss, _, _, _ = model.compute_loss(
                chars, chars_mask, strokes, strokes_mask
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


if __name__ == '__main__':
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

    model = HandwritingSynthesisNetwork(
        train_dataset.vocab_size,
        args.dec_hidden_size, args.dec_n_layers,
        args.n_mixtures_attention, args.n_mixtures_output
    ).cuda()
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if load_root and load_root.exists():
        model.load_state_dict(torch.load(load_root / 'model.pt'))

    #######################################################################
    # Dumping original data                                               #
    #######################################################################
    itr = iter(sampling_loader)
    for i in range(8):
        data = itr.__next__()
        fig = draw(
            data[2][0].numpy(),
            save_file=root / ("original_%d.png" % i)
        )
        writer.add_figure("samples/original_%d" % i, fig, 0)

    steps = 0
    for epoch in range(1, args.epochs + 1):
        print("Generating samples...")
        start = time.time()
        monitor_samples()
        print("Took %5.3f seconds to generate samples" % (time.time() - start))

        train(epoch)

        print("Testing...")
        start = time.time()
        test(epoch)
        print("Took %5.3f seconds to evaluate on test set" % (time.time() - start))

        torch.save(model.state_dict(), root / "model.pt")
