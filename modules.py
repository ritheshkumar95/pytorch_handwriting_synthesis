import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import concatenate_dict


def mixture_of_bivariate_normal_nll(
    data, log_pi, mu, log_sigma, rho, eps=1e-6
):
    x, y = data.unsqueeze(-2).unbind(-1)
    mu_1, mu_2 = mu.unbind(-1)

    log_sigma_1, log_sigma_2 = log_sigma.unbind(-1)
    sigma_1 = log_sigma_1.exp() + eps
    sigma_2 = log_sigma_2.exp() + eps

    # Compute log prob of bivariate normal distribution
    Z = torch.pow((x - mu_1) / sigma_1, 2) + torch.pow((y - mu_2) / sigma_2, 2)
    Z -= 2 * rho * ((x - mu_1) * (y - mu_2)) / (sigma_1 * sigma_2)

    log_N = -Z / (2 * (1 - rho ** 2) + eps)
    log_N -= np.log(2 * np.pi) + log_sigma_1 + log_sigma_2
    log_N -= .5 * torch.log(1 - rho ** 2 + eps)

    # Use log_sum_exp to accurately compute log prob of mixture distribution
    nll = -torch.logsumexp(log_pi + log_N, dim=-1)
    return nll


def mixture_of_bivariate_normal_sample(
    log_pi, mu, log_sigma, rho, eps=1e-6, bias=0.
):
    batch_size = log_pi.shape[0]
    ndims = log_pi.dim()

    if ndims > 2:
        # Collapse batch and seq_len dimensions
        log_pi, mu, log_sigma, rho = [
            x.reshape(-1, *x.shape[2:])
            for x in [log_pi, mu, log_sigma, rho]
        ]

    # Sample mixture index using mixture probabilities pi
    pi = log_pi.exp() * (1 + bias)
    mixture_idx = pi.multinomial(1).squeeze(1)

    # Index the correct mixture for mu, log_sigma and rho
    mu, log_sigma, rho = [
        x[torch.arange(mixture_idx.shape[0]), mixture_idx]
        for x in [mu, log_sigma, rho]
    ]

    # Calculate biased variances
    sigma = (log_sigma - bias).exp()

    # Sample from the bivariate normal distribution
    mu_1, mu_2 = mu.unbind(-1)
    sigma_1, sigma_2 = sigma.unbind(-1)
    z_1 = torch.randn_like(mu_1)
    z_2 = torch.randn_like(mu_2)

    x = mu_1 + sigma_1 * z_1
    y = mu_2 + sigma_2 * (z_2 * ((1 - rho ** 2) ** .5) + z_1 * rho)

    # Uncollapse the matrix to a tensor (if necessary)
    sample = torch.stack([x, y], 1)
    if ndims > 2:
        sample = sample.view(batch_size, -1, 2)

    return sample


class OneHotEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, arr, mask):
        shp = arr.size() + (self.vocab_size,)
        one_hot_arr = torch.zeros(shp).float().cuda()
        one_hot_arr.scatter_(-1, arr.unsqueeze(-1), 1)
        return one_hot_arr * mask.unsqueeze(-1)


class GaussianAttention(nn.Module):
    def __init__(self, hidden_size, n_mixtures, attention_multiplier=.05):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 3 * n_mixtures)

        self.n_mixtures = n_mixtures
        self.attention_multiplier = attention_multiplier

    def forward(self, h_t, k_tm1, ctx, ctx_mask):
        B, T, _ = ctx.shape
        device = ctx.device

        alpha, beta, kappa = torch.exp(self.linear(h_t))[:, None].chunk(3, dim=-1)  # (B, 1, K) each
        kappa = kappa * self.attention_multiplier + k_tm1.unsqueeze(1)

        u = torch.arange(T, dtype=torch.float32).to(device)
        u = u[None, :, None].repeat(B, 1, 1)  # (B, T, 1)
        phi = alpha * torch.exp(-beta * torch.pow(kappa - u, 2))  # (B, T, K)
        phi = phi.sum(-1) * ctx_mask
        w = (phi.unsqueeze(-1) * ctx).sum(1)

        attention_vars = {
            'alpha': alpha.squeeze(1),
            'beta': beta.squeeze(1),
            'kappa': kappa.squeeze(1),
            'phi': phi,
        }
        return w, attention_vars


class HandwritingSynthesisNetwork(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers,
        n_mixtures_attention, n_mixtures_output
    ):
        super().__init__()
        self.encoder = OneHotEncoder(vocab_size)
        self.lstm_0 = nn.LSTMCell(3 + vocab_size, hidden_size)
        self.lstm_1 = nn.LSTM(3 + vocab_size + hidden_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + vocab_size + hidden_size, hidden_size, batch_first=True)
        self.attention = GaussianAttention(hidden_size, n_mixtures_attention)
        self.fc = nn.Linear(
            hidden_size * 3, n_mixtures_output * 6 + 1
        )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_mixtures_output = n_mixtures_output

    def __init__hidden(self, bsz):
        hid_0 = torch.zeros(bsz, self.hidden_size * 2).float().cuda()
        hid_0 = hid_0.chunk(2, dim=-1)
        hid_1, hid_2 = None, None
        w_0 = torch.zeros(bsz, self.vocab_size).float().cuda()
        k_0 = torch.zeros(bsz, 1).float().cuda()
        return hid_0, hid_1, hid_2, w_0, k_0

    def __parse_outputs(self, out):
        K = self.n_mixtures_output
        mu, log_sigma, pi, rho, eos = out.split([2 * K, 2 * K, K, K, 1], -1)

        # Apply activations to constrain values in the correct range
        rho = torch.tanh(rho)
        log_pi = F.log_softmax(pi, dim=-1)
        eos = torch.sigmoid(-eos)

        mu = mu.view(mu.shape[:-1] + (K, 2))
        log_sigma = log_sigma.view(log_sigma.shape[:-1] + (K, 2))

        return log_pi, mu, log_sigma, rho, eos

    def forward(self, chars, chars_mask, strokes, strokes_mask, prev_states=None):
        # Encode the characters
        chars = self.encoder(chars, chars_mask)

        if prev_states is None:
            hid_0, hid_1, hid_2, w_t, k_t = self.__init__hidden(chars.size(0))
        else:
            hid_0, hid_1, hid_2, w_t, k_t = prev_states

        lstm_0_out = []
        attention_out = []
        monitor_vars = {'phi': [], 'alpha': [], 'beta': [], 'kappa': []}

        for x_t in strokes.unbind(1):
            hid_0 = self.lstm_0(
                torch.cat([x_t, w_t], -1),
                hid_0
            )

            w_t, vars_t = self.attention(hid_0[0], k_t, chars, chars_mask)
            k_t = vars_t['kappa']

            concatenate_dict(monitor_vars, vars_t)
            lstm_0_out.append(hid_0[0])
            attention_out.append(w_t)

        lstm_0_out = torch.stack(lstm_0_out, 1)
        attention_out = torch.stack(attention_out, 1)

        lstm_1_out, hid_1 = self.lstm_1(
            torch.cat([strokes, attention_out, lstm_0_out], -1),
            hid_1
        )

        lstm_2_out, hid_2 = self.lstm_2(
            torch.cat([strokes, attention_out, lstm_1_out], -1),
            hid_2
        )

        last_out = self.fc(
            torch.cat([lstm_0_out, lstm_1_out, lstm_2_out], -1)
        )

        output_params = self.__parse_outputs(last_out)
        monitor_vars = {x: torch.stack(y, 1) for x, y in monitor_vars.items()}
        return output_params, monitor_vars, (hid_0, hid_1, hid_2, w_t, k_t)

    def sample(self, chars, chars_mask, maxlen=1000):
        chars = self.encoder(chars, chars_mask)
        last_idx = (chars_mask.sum(-1) - 2).long()

        hid_0, hid_1, hid_2, w_t, k_t = self.__init__hidden(chars.size(0))
        x_t = torch.zeros(chars.size(0), 3).float().cuda()

        strokes = []
        monitor_vars = {'phi': [], 'kappa': [], 'alpha': [], 'beta': []}
        for i in range(maxlen):
            hid_0 = self.lstm_0(
                torch.cat([x_t, w_t], -1),
                hid_0
            )

            w_t, vars_t = self.attention(hid_0[0], k_t, chars, chars_mask)
            k_t = vars_t['kappa']

            concatenate_dict(monitor_vars, vars_t)

            _, hid_1 = self.lstm_1(
                torch.cat([x_t, w_t, hid_0[0]], 1).unsqueeze(1),
                hid_1
            )  # hid_1 - tuple of (1, batch_size, hidden_size)

            _, hid_2 = self.lstm_2(
                torch.cat([x_t, w_t, hid_1[0].squeeze(0)], 1).unsqueeze(1),
                hid_2
            )  # hid_2 - tuple of (1, batch_size, hidden_size)

            last_out = self.fc(
                torch.cat([hid_0[0], hid_1[0].squeeze(0), hid_2[0].squeeze(0)], 1)
            )
            output_params = self.__parse_outputs(last_out)

            x_t = torch.cat([
                output_params[-1].bernoulli(),
                mixture_of_bivariate_normal_sample(*output_params[:-1], bias=3.)
            ], dim=1)

            ################################################
            # Exit Condition                               #
            ################################################
            phi_t = vars_t['kappa']
            check_1 = ~torch.gt(phi_t.max(1)[1], last_idx)
            check_2 = torch.sign(phi_t.sum(1)).byte()
            is_incomplete = check_1 | check_2

            if is_incomplete.sum().item() == 0:
                break

            x_t = x_t * is_incomplete.float().unsqueeze(-1)
            strokes.append(x_t)

        monitor_vars = {x: torch.stack(y, 1) for x, y in monitor_vars.items()}
        return torch.stack(strokes, 1), monitor_vars

    def compute_loss(self, chars, chars_mask, strokes, strokes_mask, prev_states=None):
        input_strokes = strokes[:, :-1]
        input_strokes_mask = strokes_mask[:, :-1]
        output_strokes = strokes[:, 1:]

        output_params, monitor_vars, prev_states = self.forward(
            chars, chars_mask, input_strokes, input_strokes_mask,
            prev_states
        )

        stroke_loss = mixture_of_bivariate_normal_nll(
            output_strokes[:, :, 1:],
            *output_params[:-1]  # passing everything except eos param
        )
        stroke_loss = (stroke_loss * input_strokes_mask).sum(-1).mean()

        eos_loss = F.binary_cross_entropy(
            output_params[-1].squeeze(-1),
            output_strokes[:, :, 0],
            reduction='none'
        )
        eos_loss = (eos_loss * input_strokes_mask).sum(-1).mean()

        teacher_forced_sample = torch.cat([
            output_params[-1].bernoulli(),
            mixture_of_bivariate_normal_sample(*output_params[:-1], bias=3.)
        ], dim=-1)

        return stroke_loss, eos_loss, monitor_vars, prev_states, teacher_forced_sample


class HandwritingPredictionNetwork(nn.Module):
    def __init__(
        self, hidden_size, n_layers, n_mixtures_output
    ):
        super().__init__()
        self.lstm_0 = nn.LSTM(3, hidden_size, batch_first=True)
        self.lstm_1 = nn.LSTM(3 + hidden_size, hidden_size, batch_first=True)
        self.lstm_2 = nn.LSTM(3 + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(
            hidden_size * 3, n_mixtures_output * 6 + 1
        )

        self.hidden_size = hidden_size
        self.n_mixtures_output = n_mixtures_output

    def __parse_outputs(self, out):
        K = self.n_mixtures_output
        mu, log_sigma, pi, rho, eos = out.split([2 * K, 2 * K, K, K, 1], -1)

        # Apply activations to constrain values in the correct range
        rho = torch.tanh(rho)
        log_pi = F.log_softmax(pi, dim=-1)
        eos = torch.sigmoid(-eos)

        mu = mu.view(mu.shape[:-1] + (K, 2))
        log_sigma = log_sigma.view(log_sigma.shape[:-1] + (K, 2))

        return log_pi, mu, log_sigma, rho, eos

    def forward(self, strokes, strokes_mask, prev_states=None):
        if prev_states is None:
            hid_0, hid_1, hid_2 = None, None, None
        else:
            hid_0, hid_1, hid_2 = prev_states

        lstm_0_out, hid_0 = self.lstm_0(
            strokes, hid_0
        )

        lstm_1_out, hid_1 = self.lstm_1(
            torch.cat([strokes, lstm_0_out], -1),
            hid_1
        )

        lstm_2_out, hid_2 = self.lstm_2(
            torch.cat([strokes, lstm_1_out], -1),
            hid_2
        )

        last_out = self.fc(
            torch.cat([lstm_0_out, lstm_1_out, lstm_2_out], -1)
        )

        output_params = self.__parse_outputs(last_out)
        return output_params, (hid_0, hid_1, hid_2)

    def sample(self, batch_size=1, maxlen=1000):
        hid_0, hid_1, hid_2 = None, None, None
        x_t = torch.zeros(batch_size, 1, 3).float().cuda()

        strokes = []
        for i in range(maxlen):
            _, hid_0 = self.lstm_0(x_t, hid_0)

            _, hid_1 = self.lstm_1(
                torch.cat([x_t, hid_0[0]], -1),
                hid_1
            )  # hid_1 - tuple of (1, batch_size, hidden_size)

            _, hid_2 = self.lstm_2(
                torch.cat([x_t, hid_1[0]], -1),
                hid_2
            )  # hid_2 - tuple of (1, batch_size, hidden_size)

            last_out = self.fc(
                torch.cat([hid_0[0], hid_1[0], hid_2[0]], -1)
            ).squeeze(1)
            output_params = self.__parse_outputs(last_out)

            x_t = torch.cat([
                output_params[-1].bernoulli(),
                mixture_of_bivariate_normal_sample(*output_params[:-1], bias=3.)
            ], dim=1).unsqueeze(1)

            strokes.append(x_t)

        return torch.cat(strokes, 1)

    def compute_loss(self, strokes, strokes_mask, prev_states=None):
        input_strokes = strokes[:, :-1]
        input_strokes_mask = strokes_mask[:, :-1]
        output_strokes = strokes[:, 1:]

        output_params, prev_states = self.forward(
            input_strokes, input_strokes_mask,
            prev_states
        )

        stroke_loss = mixture_of_bivariate_normal_nll(
            output_strokes[:, :, 1:],
            *output_params[:-1]  # passing everything except eos param
        )
        stroke_loss = (stroke_loss * input_strokes_mask).sum(-1).mean()

        eos_loss = F.binary_cross_entropy(
            output_params[-1].squeeze(-1),
            output_strokes[:, :, 0],
            reduction='none'
        )
        eos_loss = (eos_loss * input_strokes_mask).sum(-1).mean()

        teacher_forced_sample = torch.cat([
            output_params[-1].bernoulli(),
            mixture_of_bivariate_normal_sample(*output_params[:-1], bias=3.)
        ], dim=-1)

        return stroke_loss, eos_loss, prev_states, teacher_forced_sample


if __name__ == '__main__':
    vocab_size = 60
    hidden_size = 400
    n_layers = 3
    K_att = 6
    K_out = 20

    model = HandwritingSynthesisNetwork(
        vocab_size, hidden_size, n_layers,
        K_att, K_out
    ).cuda()

    chars = torch.randint(0, vocab_size, (4, 50)).cuda()
    chars_mask = torch.ones_like(chars).float()

    strokes = torch.randn(4, 300, 3).cuda()
    strokes_mask = torch.ones(4, 300).cuda()

    loss = model.compute_loss(chars, chars_mask, strokes, strokes_mask)
    print(loss)

    out = model.sample(chars, chars_mask)
    print(out[0].shape)
