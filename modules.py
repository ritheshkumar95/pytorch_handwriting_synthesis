import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.distribution import Distribution


def make_contiguous(hid_t):
    return (hid_t[0].contiguous(), hid_t[1].contiguous())


class MixtureOfBivariateNormal(Distribution):
    def __init__(self, log_pi, mu, log_sigma, rho):
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

    def log_prob(self, x):
        t = (x - self.mu) / self.log_sigma.exp()
        Z = (t ** 2).sum(-1) - 2 * self.rho * torch.prod(t, -1)

        num = -Z / (2 * (1 - self.rho ** 2))
        denom = np.log(2 * np.pi) + self.log_sigma.sum(-1) + .5 * torch.log(1 - self.rho ** 2)
        log_N = num - denom
        log_prob = torch.logsumexp(self.log_pi + log_N, dim=-1)
        return -log_prob

    def sample(self):
        index = self.log_pi.exp().multinomial(1).squeeze(1)
        mu = self.mu[torch.arange(index.shape[0]), index]
        sigma = self.log_sigma.exp()[torch.arange(index.shape[0]), index]
        rho = self.rho[torch.arange(index.shape[0]), index]

        mu1, mu2 = mu.unbind(-1)
        sigma1, sigma2 = sigma.unbind(-1)
        z1 = torch.randn_like(mu1)
        z2 = torch.randn_like(mu2)

        x1 = mu1 + sigma1 * z1
        mult = z2 * ((1 - rho ** 2) ** .5) + z1 * rho
        x2 = mu2 + sigma2 * mult
        return torch.stack([x1, x2], 1)


class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)

    def forward(self, src, mask):
        return self.emb(src)


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)

        self.rnn = nn.LSTM(
            emb_size, hidden_size, n_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, src, mask):
        lengths = mask.sum(-1)

        src = self.emb(src)
        src = pack_padded_sequence(src, lengths, batch_first=True)
        out, _ = self.rnn(src)
        out = pad_packed_sequence(out, batch_first=True)[0]
        return out


class GaussianAttention(nn.Module):
    def __init__(self, hidden_size, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.linear = nn.Linear(hidden_size, 3 * n_mixtures)

    def forward(self, h_t, ctx, mask):
        B, T, _ = ctx.shape
        device = ctx.device

        alpha, beta, kappa = torch.exp(self.linear(h_t))[:, :, None].chunk(3, dim=-1)  # (B, t, 1, K) each
        kappa *= .2
        kappa = kappa.cumsum(1)

        u = torch.arange(T, dtype=torch.float32).to(device)
        u = u[None, None, :, None].repeat(B, 1, 1, 1)  # (B, 1, T, 1)
        phi = alpha * torch.exp(-beta * torch.pow(kappa - u, 2))  # (B, t, T, K)
        phi = phi.sum(-1)

        monitor = {
            'alpha': alpha.squeeze(1),
            'beta': beta.squeeze(1),
            'kappa': kappa.squeeze(1),
            'phi': phi,
        }
        return (phi.unsqueeze(-1) * ctx[:, None] * mask[:, None, :, None]).sum(2), monitor


class RNNDecoder(nn.Module):
    def __init__(
        self, enc_size, hidden_size, n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        self.layer_0 = nn.LSTM(3, hidden_size, batch_first=True)
        self.layer_1 = nn.LSTM(3 + enc_size + hidden_size, hidden_size, batch_first=True)
        self.layer_2 = nn.LSTM(3 + enc_size + hidden_size, hidden_size, batch_first=True)
        self.attention = GaussianAttention(hidden_size, n_mixtures_attention)
        self.output = nn.Linear(
            hidden_size * 3, n_mixtures_output * 6 + 1
        )

        self.h_0 = nn.Parameter(torch.randn(n_layers, hidden_size * 2))

        self.hidden_size = hidden_size
        self.enc_size = enc_size
        self.n_mixtures_attention = n_mixtures_attention

    def forward(self, strokes, context, context_mask, prev_states=None):
        bsz = strokes.size(0)

        if prev_states is None:
            hidden = self.h_0[:, None].repeat(1, bsz, 1)  # (L, B, D)
            hid_0_tm1, hid_1_tm1, hid_2_tm1 = [
                make_contiguous(x.chunk(2, dim=-1)) for x in hidden.split(1, dim=0)
            ]
        else:
            hid_0_tm1, hid_1_tm1, hid_2_tm1 = prev_states

        layer_0_h_t, hid_0_tm1 = self.layer_0(strokes, hid_0_tm1)

        w_t, stats = self.attention(layer_0_h_t, context, context_mask)

        layer_1_h_t, hid_1_tm1 = self.layer_1(
            torch.cat([strokes, layer_0_h_t, w_t], 2),
            hid_1_tm1
        )

        layer_2_h_t, hid_2_tm1 = self.layer_2(
            torch.cat([strokes, layer_1_h_t, w_t], 2),
            hid_2_tm1
        )

        outputs = self.output(
            torch.cat([layer_0_h_t, layer_1_h_t, layer_2_h_t], -1)
        )

        return outputs, stats, (hid_0_tm1, hid_1_tm1, hid_2_tm1)


class Seq2Seq(nn.Module):
    def __init__(
        self, vocab_size, enc_emb_size, enc_hidden_size, enc_n_layers,
        dec_hidden_size, dec_n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        # self.enc = RNNEncoder(vocab_size, enc_emb_size, enc_hidden_size // 2, enc_n_layers)
        self.enc = SimpleEncoder(vocab_size, enc_emb_size)
        self.dec = RNNDecoder(
            enc_emb_size, dec_hidden_size, dec_n_layers,
            n_mixtures_attention, n_mixtures_output
        )
        self.n_mixtures_attention = n_mixtures_attention
        self.n_mixtures_output = n_mixtures_output

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
        stroke_loss = dist.log_prob(strokes[:, 1:, :2].unsqueeze(-2))
        eos_loss = F.binary_cross_entropy_with_logits(
            eos.squeeze(-1), strokes[:, 1:, -1], reduction='none'
        )

        if mask_loss:
            mask = strokes_mask[:, 1:]
            stroke_loss = (stroke_loss * mask).sum() / mask.sum()
            eos_loss = (eos_loss * mask).sum() / mask.sum()
            return stroke_loss, eos_loss, att, prev_states
        else:
            return stroke_loss.mean(), eos_loss.mean(), att, prev_states

    def sample(self, chars, chars_mask, maxlen=1000):
        K = self.n_mixtures_output

        ctx = self.enc(chars, chars_mask) * chars_mask.unsqueeze(-1)
        x_t = torch.zeros(ctx.size(0), 1, 3).float().cuda()
        prev_states = None
        strokes = []
        for i in range(maxlen):
            strokes.append(x_t)
            out, _, prev_states = self.dec(x_t, ctx, chars_mask, prev_states)

            mu, log_sigma, pi, rho, eos = out.squeeze(1).split(
                [2 * K, 2 * K, K, K, 1], dim=-1
            )
            rho = torch.tanh(rho)
            log_pi = F.log_softmax(pi, dim=-1)
            mu = mu.view(-1, K, 2)  # (B, K, 2)
            log_sigma = log_sigma.view(-1, K, 2) - np.log(5)  # (B, K, 2)

            dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)
            x_t = torch.cat([
                dist.sample(),
                torch.sigmoid(eos).bernoulli(),
            ], dim=1).unsqueeze(1)

        return torch.cat(strokes, 1)


if __name__ == '__main__':
    vocab_size = 60
    emb_size = 128
    enc_hidden_size = 256
    enc_n_layers = 1
    dec_hidden_size = 400
    dec_n_layers = 3
    K_att = 10
    K_out = 20

    model = Seq2Seq(
        vocab_size, emb_size, enc_hidden_size, enc_n_layers,
        dec_hidden_size, dec_n_layers,
        K_att, K_out
    ).cuda()
    chars = torch.randint(0, vocab_size, (16, 50)).cuda()
    chars_mask = torch.ones_like(chars).float()
    strokes = torch.randn(16, 300, 3).cuda()
    strokes_mask = torch.ones(16, 300).cuda()

    loss = model(strokes, strokes_mask, chars, chars_mask)
    print(loss)

    out = model.sample(chars, chars_mask)
    print(out.shape)
