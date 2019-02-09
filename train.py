from data.utils import draw, alphabet
from dataset import DataLoader
from modules import Generator, Discriminator
import matplotlib.pyplot as plt

import pickle
import numpy as np
import time
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tensorboardX import SummaryWriter


def train(epoch):
    global steps
    costs = []
    itr = loader.create_iterator(
        'train', batch_size=args.batch_size, mod_size=2 ** args.n_downsampling
    )
    start_time = time.time()
    for iterno, (spkrs, chars, chars_mask, strokes, strokes_mask) in enumerate(itr):
        x_t = strokes.transpose(1, 2)
        x_pred_t = netG(x_t.size(-1), spkrs, chars, chars_mask)

        ####################################################
        # Train Discriminator
        ####################################################
        D_fake_det = netD(x_pred_t.detach())
        D_real = netD(x_t)

        loss_D = 0
        for scale in D_fake_det:
            loss_D += F.mse_loss(scale[-1], torch.zeros_like(scale[-1]))
        for scale in D_real:
            loss_D += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

        netD.zero_grad()
        loss_D.backward()
        optD.step()

        ####################################################
        # Train Generator
        ####################################################
        D_fake = netD(x_pred_t)

        loss_G = 0
        for scale in D_fake:
            loss_G += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

        loss_feat = 0
        feat_weights = 4.0 / (args.n_layers_D + 1)
        D_weights = 1.0 / args.num_D
        for i in range(args.num_D):
            for j in range(len(D_fake[i]) - 1):
                fm = F.l1_loss(D_fake[i][j], D_real[i][j].detach())
                wt = D_weights * feat_weights
                loss_feat += wt * fm

        eos_loss = F.binary_cross_entropy_with_logits(
            x_pred_t[:, -1], x_t[:, -1]
        )

        netG.zero_grad()
        (loss_G + eos_loss + args.lambda_feat * loss_feat).backward()
        optG.step()

        ####################################################
        costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), eos_loss.item()])

        writer.add_scalar("loss_D/train", costs[-1][0], steps)
        writer.add_scalar("loss_G/train", costs[-1][1], steps)
        writer.add_scalar("loss_feat/train", costs[-1][2], steps)
        writer.add_scalar("loss_eos/train", costs[-1][3], steps)
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
            with torch.no_grad():
                spkrs, chars, chars_mask, stk, stk_mask = test_data
                out = netG(stk.size(1), spkrs, chars, chars_mask).detach().cpu().numpy()
                out = out.transpose(0, 2, 1)
                for i in range(8):
                    fig = draw(out[i], save_file=root / ("generated_%d.png" % i))
                    writer.add_figure("samples/generated_%d" % i, fig, steps)
            print("Took %5.4fs to generate samples" % (time.time() - st))
            print("-" * 100)


def test(epoch):
    global steps
    costs = []
    itr = loader.create_iterator(
        'test', batch_size=args.batch_size, mod_size=2 ** args.n_downsampling
    )
    start_time = time.time()

    with torch.no_grad():
        for iterno, (spkrs, chars, chars_mask, strokes, strokes_mask) in enumerate(itr):
            x_t = strokes.transpose(1, 2)
            x_pred_t = netG(x_t.size(-1), spkrs, chars, chars_mask)

            ####################################################
            # Discriminator
            ####################################################
            D_fake_det = netD(x_pred_t.detach())
            D_real = netD(x_t)

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.mse_loss(scale[-1], torch.zeros_like(scale[-1]))
            for scale in D_real:
                loss_D += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

            ####################################################
            # Generator
            ####################################################
            D_fake = netD(x_pred_t)

            loss_G = 0
            for scale in D_fake:
                loss_G += F.mse_loss(scale[-1], torch.ones_like(scale[-1]))

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    fm = F.l1_loss(D_fake[i][j], D_real[i][j].detach())
                    wt = D_weights * feat_weights
                    loss_feat += wt * fm

            eos_loss = F.binary_cross_entropy_with_logits(
                x_pred_t[:, -1], x_t[:, -1]
            )

            ####################################################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), eos_loss.item()])

    loss_D, loss_G, loss_feat, loss_eos = np.asarray(costs).mean(0)
    writer.add_scalar("loss_D/test", loss_D, steps)
    writer.add_scalar("loss_G/test", loss_G, steps)
    writer.add_scalar("loss_feat/test", loss_feat, steps)
    writer.add_scalar("loss_eos/test", loss_eos, steps)

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

    parser.add_argument("--enc_emb_size", type=int, default=128)
    parser.add_argument("--enc_hidden_size", type=int, default=256)
    parser.add_argument("--enc_n_layers", type=int, default=1)

    parser.add_argument("--n_spkrs", type=int, default=221)
    parser.add_argument("--spkr_size", type=int, default=256)

    parser.add_argument("--att_size", type=int, default=256)
    parser.add_argument("--att_n_filters", type=int, default=32)
    parser.add_argument("--att_kernel_size", type=int, default=11)
    parser.add_argument("--rnn_size", type=int, default=512)
    parser.add_argument("--aligner_size", type=int, default=512)

    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_downsampling", type=int, default=3)
    parser.add_argument("--n_residual_blocks", type=int, default=9)

    parser.add_argument("--ndf", type=int, default=32)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsampling_factor", type=int, default=2)

    parser.add_argument("--path", default='./data/processed')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_feat", type=float, default=10)

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

netG = Generator(
    args.vocab_size, args.enc_emb_size, args.enc_hidden_size, args.enc_n_layers, args.n_spkrs, args.spkr_size,
    args.rnn_size, args.aligner_size, args.att_size, args.att_n_filters, args.att_kernel_size,
    3, args.ngf, args.n_downsampling, args.n_residual_blocks
).cuda()
netD = Discriminator(
    3, args.num_D, args.ndf, args.n_layers_D, args.downsampling_factor
).cuda()

print(netG)
print(netD)

optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9))
optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.9))

if load_root and load_root.exists():
    netG.load_state_dict(torch.load(load_root / 'netG.pt'))
    netD.load_state_dict(torch.load(load_root / 'netD.pt'))

loader = DataLoader(args.path)
#######################################################################
# Dumping original data
#######################################################################
itr = loader.create_iterator('test', batch_size=8)
test_data = itr.__next__()
for i in range(8):
    fig = draw(test_data[3][i].cpu().numpy(), save_file=root / ("original_%d.png" % i))
    writer.add_figure("samples/original_%d" % i, fig, 0)

costs = []
start = time.time()
steps = 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    torch.save(netG.state_dict(), root / "netG.pt")
    torch.save(netD.state_dict(), root / "netD.pt")
