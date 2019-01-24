import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.distribution import Distribution


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


class RNNDecoder(nn.Module):
    def __init__(
        self, hidden_size, n_layers, n_mixtures_output, dropout
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            3, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output = nn.Linear(
            hidden_size, n_mixtures_output * 6 + 1
        )
        self.n_mixtures_output = n_mixtures_output

    def forward(self, strokes, prev_hidden=None):
        out, prev_hidden = self.rnn(strokes, prev_hidden)
        out = self.output(self.drop(out))
        return out, prev_hidden

    def score(self, strokes, mask):
        K = self.n_mixtures_output
        out = self.forward(strokes[:, :-1])[0]

        mu, log_sigma, pi, rho, eos = out.split([2 * K, 2 * K, K, K, 1], -1)

        rho = torch.tanh(rho)
        log_pi = F.log_softmax(pi, dim=-1)

        mu = mu.view(mu.shape[:2] + (K, 2))  # (B, T, K, 2)
        log_sigma = log_sigma.view(log_sigma.shape[:2] + (K, 2))  # (B, T, K, 2)

        output_dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)
        stroke_loss = output_dist.log_prob(strokes[:, 1:, :2].unsqueeze(-2))
        eos_loss = F.binary_cross_entropy_with_logits(
            eos.squeeze(-1), strokes[:, 1:, -1], reduction='none'
        )
        mask = mask[:, 1:]
        stroke_loss = (stroke_loss * mask).sum() / mask.sum()
        eos_loss = (eos_loss * mask).sum() / mask.sum()
        return stroke_loss, eos_loss

    def sample(self, batch_size=8, maxlen=600):
        K = self.n_mixtures_output
        x_t = torch.zeros(batch_size, 1, 3).float().cuda()
        prev_hidden = None
        strokes = []
        for i in range(maxlen):
            strokes.append(x_t)
            out, prev_hidden = self.forward(x_t, prev_hidden)

            mu, log_sigma, pi, rho, eos = out.squeeze(1).split(
                [2 * K, 2 * K, K, K, 1], dim=-1
            )
            rho = torch.tanh(rho)
            log_pi = F.log_softmax(pi, dim=-1)
            mu = mu.view(-1, K, 2)  # (B, K, 2)
            log_sigma = log_sigma.view(-1, K, 2)  # (B, K, 2)

            output_dist = MixtureOfBivariateNormal(log_pi, mu, log_sigma, rho)
            x_t = torch.cat([
                output_dist.sample(),
                torch.sigmoid(eos).bernoulli(),
            ], dim=1).unsqueeze(1)

        return torch.cat(strokes, 1)


if __name__ == '__main__':
    vocab_size = 60
    emb_size = 128
    hidden_size = 256
    n_layers = 3
    K_out = 20
    dropout = .1

    model = RNNDecoder(hidden_size, n_layers, K_out, dropout).cuda()
    strokes = torch.randn(16, 300, 3).cuda()

    loss = model.score(strokes)
    print(loss)

    out = model.sample()
    print(out.shape)
