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

    def sort(self, src, mask):
        lengths, idx = torch.sort(mask.sum(-1), 0, descending=True)
        return src[idx], lengths, idx

    def unsort(self, src, idx):
        idx = torch.argsort(idx, 0)
        return src[idx]

    def forward(self, src, mask):
        # src, lengths, idx = self.sort(src, mask)
        lengths = mask.sum(-1)

        src = self.emb(src)
        src = pack_padded_sequence(src, lengths, batch_first=True)
        out, _ = self.rnn(src)
        out = pad_packed_sequence(out, batch_first=True)[0]

        # out = self.unsort(out, idx)
        return out


class GaussianAttention(nn.Module):
    def __init__(self, hidden_size, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.linear = nn.Linear(hidden_size, 3 * n_mixtures)

    def forward(self, h_t, k_tm1, ctx, mask):
        B, T, _ = ctx.shape
        device = ctx.device

        alpha, beta, kappa = torch.exp(self.linear(h_t))[:, None].chunk(3, dim=-1)  # (B, 1, K) each
        kappa += k_tm1

        u = torch.arange(T, dtype=torch.float32).to(device)
        u = u[None, :, None].repeat(B, 1, 1)  # (B, T, 1)
        phi = alpha * torch.exp(-torch.pow(kappa - u, 2) / beta)  # (B, T, K)
        phi = phi.sum(-1, keepdim=True)

        monitor = {
            'alpha': alpha,
            'beta': beta,
            'kappa': kappa,
            'phi': phi,
        }

        return (phi * ctx * mask.unsqueeze(-1)).sum(1), monitor


class RNNDecoder(nn.Module):
    def __init__(
        self, enc_size, hidden_size, n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        self.layer_0 = nn.LSTMCell(
            3 + enc_size, hidden_size
        )
        self.layer_n = nn.ModuleList([
            nn.LSTMCell(3 + enc_size + hidden_size, hidden_size)
            for i in range(n_layers - 1)
        ])
        self.attention = GaussianAttention(hidden_size, n_mixtures_attention)
        self.output = nn.Linear(
            hidden_size * n_layers, n_mixtures_output * 6 + 1
        )
        self.h_0 = nn.Parameter(torch.randn(n_layers, hidden_size))
        self.c_0 = nn.Parameter(torch.randn(n_layers, hidden_size))
        self.w_0 = nn.Parameter(torch.randn(enc_size))
        self.register_buffer(
            'k_0', torch.zeros(n_mixtures_attention)
        )

    def forward(self, strokes, context, context_mask, prev_states=None):
        bsz = strokes.size(0)

        if prev_states is None:
            h_tm1 = list(self.h_0[None].repeat(bsz, 1, 1).unbind(1))
            c_tm1 = list(self.c_0[None].repeat(bsz, 1, 1).unbind(1))
            w_tm1 = self.w_0[None].repeat(bsz, 1)
            k_tm1 = self.k_0[None, None].repeat(bsz, 1, 1)
        else:
            h_tm1, c_tm1, w_tm1, k_tm1 = prev_states

        outputs = []
        attention = {'phi': [], 'alpha': [], 'beta': [], 'kappa': []}
        for i, x_t in enumerate(strokes.unbind(1)):
            h_tm1[0], c_tm1[0] = self.layer_0(
                torch.cat([x_t, w_tm1], 1),
                (h_tm1[0], c_tm1[0])
            )

            w_tm1, monitor = self.attention(h_tm1[0], k_tm1, context, context_mask)
            k_tm1 = monitor['kappa']
            append_dict(attention, monitor)

            for j, layer in enumerate(self.layer_n):
                h_tm1[j+1], c_tm1[j+1] = layer(
                    torch.cat([x_t, w_tm1, h_tm1[j]], 1),
                    (h_tm1[j+1], c_tm1[j+1])
                )

            out = self.output(torch.cat(h_tm1, 1))
            outputs.append(out)

        attention = {key: torch.stack(value, 1) for key, value in attention.items()}
        return torch.stack(outputs, 1), attention, (h_tm1, c_tm1, w_tm1, k_tm1)


class Seq2Seq(nn.Module):
    def __init__(
        self, vocab_size, enc_emb_size, enc_hidden_size, enc_n_layers,
        dec_hidden_size, dec_n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        # self.enc = SimpleEncoder(vocab_size, enc_emb_size)
        self.enc = RNNEncoder(vocab_size, enc_emb_size, enc_hidden_size // 2, enc_n_layers)
        self.dec = RNNDecoder(
            enc_hidden_size, dec_hidden_size, dec_n_layers,
            n_mixtures_attention, n_mixtures_output
        )
        self.n_mixtures_attention = n_mixtures_attention
        self.n_mixtures_output = n_mixtures_output

    def forward(self, strokes, strokes_mask, chars, chars_mask, prev_states=None):
        K = self.n_mixtures_output

        ctx = self.enc(chars, chars_mask) * chars_mask.unsqueeze(-1)
        out, att, prev_states = self.dec(strokes[:, :-1], ctx, chars_mask, prev_states)

        mu, log_sigma, pi, rho, eos = out.split([2 * K, 2 * K, K, K, 1], -1)
        rho = torch.tanh(rho) * .99
        log_pi = F.log_softmax(pi, dim=-1)

        mu = mu.view(mu.shape[:2] + (K, 2))  # (B, T, K, 2)
        log_sigma = log_sigma.view(log_sigma.shape[:2] + (K, 2))  # (B, T, K, 2)

        dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)
        stroke_loss = dist.log_prob(strokes[:, 1:, :2].unsqueeze(-2))
        eos_loss = F.binary_cross_entropy_with_logits(
            eos.squeeze(-1), strokes[:, 1:, -1], reduction='none'
        )
        mask = strokes_mask[:, 1:]
        stroke_loss = (stroke_loss * mask).sum() / mask.sum()
        eos_loss = (eos_loss * mask).sum() / mask.sum()
        return stroke_loss, eos_loss, att, prev_states

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
            rho = torch.tanh(rho) * .99
            log_pi = F.log_softmax(pi, dim=-1)
            mu = mu.view(-1, K, 2)  # (B, K, 2)
            log_sigma = log_sigma.view(-1, K, 2)  # (B, K, 2)

            dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)
            x_t = torch.cat([
                dist.sample(),
                torch.sigmoid(eos).bernoulli(),
            ], dim=1).unsqueeze(1)

        return torch.cat(strokes, 1)


if __name__ == '__main__':
    vocab_size = 60
    emb_size = 128
    hidden_size = 256
    n_layers = 3
    K_att = 10
    K_out = 20

    model = Seq2Seq(vocab_size, emb_size, hidden_size, n_layers, K_att, K_out).cuda()
    chars = torch.randint(0, vocab_size, (16, 50)).cuda()
    chars_mask = torch.ones_like(chars).float()
    strokes = torch.randn(16, 300, 3).cuda()

    loss = model(strokes, chars, chars_mask)
    print(loss)

    out = model.sample(chars, chars_mask)
    print(out.shape)
