import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.distribution import Distribution


def append_dict(main_dict, new_dict):
    for key in main_dict.keys():
        main_dict[key] += [new_dict[key]]


class MixtureOfBivariateNormal(Distribution):
    def __init__(self, log_pi, mu, log_sigma, rho, eps=1e-6, bias=0.):
        '''
        Mixture of bivariate normal distribution
        Args:
            mu, sigma - (B, T, K, 2)
            rho - (B, T, K)
            log_pi - (B, T, K)
        '''
        super().__init__()
        self.log_pi = log_pi
        self.mu = mu
        self.log_sigma = log_sigma
        self.rho = rho

        self.bias = bias
        self.eps = eps

    def log_prob(self, x):
        t = (x - self.mu) / (self.log_sigma.exp() + self.eps)
        Z = (t ** 2).sum(-1) - 2 * self.rho * torch.prod(t, -1)

        num = -Z / (2 * (1 - self.rho ** 2) + self.eps)
        denom = np.log(2 * np.pi) + self.log_sigma.sum(-1) + .5 * torch.log(1 - self.rho ** 2 + self.eps)
        log_N = num - denom
        log_prob = torch.logsumexp(self.log_pi + log_N, dim=-1)
        return -log_prob

    def sample(self):
        flag = False
        if self.log_pi.dim() == 3:
            B, T, N = self.log_pi.shape
            flag = True
            self.log_pi = self.log_pi.view(-1, N)
            self.mu = self.mu.contiguous().view(-1, N, 2)
            self.log_sigma = self.log_sigma.contiguous().view(-1, N, 2)
            self.rho = self.rho.contiguous().view(-1, N)

        index = (
            self.log_pi.exp() * (1 + self.bias)
        ).multinomial(1).squeeze(1)
        mu = self.mu[torch.arange(index.shape[0]), index]
        sigma = (self.log_sigma - self.bias).exp()[torch.arange(index.shape[0]), index]
        rho = self.rho[torch.arange(index.shape[0]), index]

        mu1, mu2 = mu.unbind(-1)
        sigma1, sigma2 = sigma.unbind(-1)
        z1 = torch.randn_like(mu1)
        z2 = torch.randn_like(mu2)

        x1 = mu1 + sigma1 * z1
        mult = z2 * ((1 - rho ** 2) ** .5) + z1 * rho
        x2 = mu2 + sigma2 * mult
        out = torch.stack([x1, x2], 1)
        if flag:
            out = out.view(B, T, 2)
        return out


class OneHotEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, arr, mask):
        shp = arr.size() + (self.vocab_size,)
        one_hot_arr = torch.zeros(shp).float().cuda()
        one_hot_arr.scatter_(-1, arr.unsqueeze(-1), 1)
        return one_hot_arr


class GaussianAttention(nn.Module):
    def __init__(self, hidden_size, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.linear = nn.Linear(hidden_size, 3 * n_mixtures)

    def forward(self, h_t, k_tm1, ctx, ctx_mask):
        B, T, _ = ctx.shape
        device = ctx.device

        alpha, beta, kappa = torch.exp(self.linear(h_t))[:, None].chunk(3, dim=-1)  # (B, 1, K) each
        kappa = kappa + k_tm1.unsqueeze(1)

        u = torch.arange(T, dtype=torch.float32).to(device)
        u = u[None, :, None].repeat(B, 1, 1)  # (B, T, 1)
        phi = alpha * torch.exp(-beta * torch.pow(kappa - u, 2))  # (B, T, K)
        phi = phi.sum(-1) * ctx_mask

        monitor = {
            'alpha': alpha.squeeze(1),
            'beta': beta.squeeze(1),
            'kappa': kappa.squeeze(1),
            'phi': phi,
        }
        return (phi.unsqueeze(-1) * ctx).sum(1), monitor


class RNNDecoder(nn.Module):
    def __init__(
        self, enc_size, hidden_size, n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        self.lstm_0 = nn.LSTMCell(3 + enc_size, hidden_size)
        self.lstm_1 = nn.LSTM(3 + enc_size + hidden_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + enc_size + hidden_size, hidden_size, batch_first=True)
        self.attention = GaussianAttention(hidden_size, n_mixtures_attention)
        self.fc = nn.Linear(
            hidden_size * 3, n_mixtures_output * 6 + 1
        )

        self.hidden_size = hidden_size
        self.enc_size = enc_size
        self.n_mixtures_attention = n_mixtures_attention

    def __init__hidden(self, bsz):
        hid_0 = torch.zeros(bsz, self.hidden_size * 2).float().cuda()
        hid_0 = hid_0.chunk(2, dim=-1)
        hiddens = (hid_0, None, None)
        w_0 = torch.zeros(bsz, self.enc_size).float().cuda()
        k_0 = torch.zeros(bsz, 1).float().cuda()
        return hiddens, w_0, k_0

    def forward(self, strokes, context, context_mask, prev_states=None):
        bsz = strokes.size(0)

        if prev_states is None:
            [hid_0, hid_1, hid_2], w_t, k_t = self.__init__hidden(bsz)
        else:
            [hid_0, hid_1, hid_2], w_t, k_t = prev_states

        outputs = []
        monitor = {'phi': [], 'kappa': [], 'alpha': [], 'beta': []}
        for x_t in strokes.unbind(1):
            hid_0 = self.lstm_0(
                torch.cat([x_t, w_t], 1),
                hid_0
            )

            w_t, stats = self.attention(hid_0[0], k_t, context, context_mask)
            k_t = stats['kappa']

            outputs.append([hid_0[0], w_t])
            append_dict(monitor, stats)

        hid_0_arr, w_t_arr = zip(*outputs)
        hid_0_arr = torch.stack(hid_0_arr, 1)
        w_t_arr = torch.stack(w_t_arr, 1)
        hid_1_arr, hid_1 = self.lstm_1(
            torch.cat([strokes, hid_0_arr, w_t_arr], -1),
            hid_1
        )

        hid_2_arr, hid_2 = self.lstm_2(
            torch.cat([strokes, hid_1_arr, w_t_arr], -1),
            hid_2
        )

        outputs = self.fc(
            torch.cat([hid_0_arr, hid_1_arr, hid_2_arr], -1)
        )

        monitor = {x: torch.stack(y, 1) for x, y in monitor.items()}
        return outputs, monitor, ([hid_0, hid_1, hid_2], w_t, k_t)


class Seq2Seq(nn.Module):
    def __init__(
        self, vocab_size, dec_hidden_size, dec_n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        self.enc = OneHotEncoder(vocab_size)
        self.dec = RNNDecoder(
            vocab_size, dec_hidden_size, dec_n_layers,
            n_mixtures_attention, n_mixtures_output
        )
        self.n_mixtures_attention = n_mixtures_attention
        self.n_mixtures_output = n_mixtures_output

        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_normal_(param)
        #     elif 'phi' in name:
        #             torch.nn.init.constant_(param, -2.)

    def forward(self, strokes, strokes_mask, chars, chars_mask, prev_states=None, mask_loss=True):
        K = self.n_mixtures_output

        ctx = self.enc(chars, chars_mask) * chars_mask.unsqueeze(-1)
        out, att, prev_states = self.dec(strokes[:, :-1], ctx, chars_mask, prev_states)

        mu, log_sigma, pi, rho, eos = out.split([2 * K, 2 * K, K, K, 1], -1)
        rho = torch.tanh(rho)
        log_pi = F.log_softmax(pi, dim=-1)

        mu = mu.view(mu.shape[:2] + (K, 2))  # (B, T, K, 2)
        log_sigma = log_sigma.view(log_sigma.shape[:2] + (K, 2))  # (B, T, K, 2)

        dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)

        stroke_loss = dist.log_prob(strokes[:, 1:, 1:].unsqueeze(-2))
        eos_loss = F.binary_cross_entropy_with_logits(
            -eos.squeeze(-1), strokes[:, 1:, 0], reduction='none'
        )

        samp = torch.cat([
            torch.sigmoid(-eos).bernoulli(),
            dist.sample(),
        ], dim=-1)

        if mask_loss:
            mask = strokes_mask[:, 1:]
            stroke_loss = (stroke_loss * mask).sum(-1).mean()
            eos_loss = (eos_loss * mask).sum(-1).mean()
            return stroke_loss, eos_loss, att, prev_states, samp
        else:
            return stroke_loss.mean(), eos_loss.mean(), att, prev_states, samp

    def sample(self, chars, chars_mask, maxlen=1000):
        K = self.n_mixtures_output

        ctx = self.enc(chars, chars_mask) * chars_mask.unsqueeze(-1)
        max_char_idx = (chars_mask.sum(-1) - 1).long()
        # print(max_char_idx)
        # print(chars.shape)
        # input()
        prev_max = None

        x_t = torch.zeros(ctx.size(0), 1, 3).float().cuda()
        prev_states = None
        strokes = []
        for i in range(maxlen):
            strokes.append(x_t)
            out, monitor, prev_states = self.dec(x_t, ctx, chars_mask, prev_states)
            phi = monitor['phi'].squeeze(1)
            is_incomplete = 1 - torch.gt(phi.max(1)[1], max_char_idx).float()
            # print(phi.max(1)[1].item())

            mu, log_sigma, pi, rho, eos = out.squeeze(1).split(
                [2 * K, 2 * K, K, K, 1], dim=-1
            )
            rho = torch.tanh(rho)
            log_pi = F.log_softmax(pi, dim=-1)
            mu = mu.view(-1, K, 2)  # (B, K, 2)
            log_sigma = log_sigma.view(-1, K, 2)  # (B, K, 2)

            dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho, bias=3.)
            x_t = torch.cat([
                torch.sigmoid(-eos).bernoulli(),
                dist.sample(),
                ], dim=1).unsqueeze(1) * is_incomplete[:, None, None]

            if is_incomplete.sum().item() == 0 or phi.sum().item() == 0:
                # print('Breaking out of sampling early!')
                break

        return torch.cat(strokes, 1)


if __name__ == '__main__':
    vocab_size = 60
    dec_hidden_size = 400
    dec_n_layers = 3
    K_att = 6
    K_out = 20

    model = Seq2Seq(
        vocab_size, dec_hidden_size, dec_n_layers,
        K_att, K_out
    ).cuda()
    chars = torch.randint(0, vocab_size, (4, 50)).cuda()
    chars_mask = torch.ones_like(chars).float()
    strokes = torch.randn(4, 300, 3).cuda()
    strokes_mask = torch.ones(4, 300).cuda()

    loss = model(strokes, strokes_mask, chars, chars_mask)
    print(loss)

    out = model.sample(chars, chars_mask)
    print(out.shape)
