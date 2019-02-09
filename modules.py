import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = nn.Conv1d(
            2, attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding, bias=False
        )
        self.location_dense = nn.Linear(
            attention_n_filters, attention_dim,
            bias=False
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(1 - mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


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


class Aligner(nn.Module):
    def __init__(self, n_spkrs, spkr_dim, enc_dim, out_dim, rnn_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.attention_layer = Attention(
            rnn_dim, enc_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size
        )
        self.spkr_emb = nn.Embedding(n_spkrs, spkr_dim)
        self.rnn = nn.LSTMCell(enc_dim + spkr_dim, rnn_dim)
        self.linear_projection = nn.Linear(rnn_dim + enc_dim, out_dim)

        self.rnn_dim = rnn_dim

    def forward(self, output_timesteps, spkr, memory, mask):
        B, T, D = memory.size()
        processed_memory = self.attention_layer.memory_layer(memory)
        spkr = self.spkr_emb(spkr)

        # Initial hidden states
        alpha_tm1 = torch.zeros(B, T).cuda()
        alpha_cumulative = torch.zeros(B, T).cuda()
        h_tm1 = torch.zeros(B, self.rnn_dim * 2).cuda().chunk(2, dim=-1)
        w_tm1 = torch.zeros(B, D).cuda()

        outputs = []
        for i in range(output_timesteps):
            h_tm1 = self.rnn(torch.cat([w_tm1, spkr], 1), h_tm1)
            alpha_cat = torch.stack([alpha_tm1, alpha_cumulative], dim=1)
            w_tm1, alpha_tm1 = self.attention_layer(
                h_tm1[0], memory, processed_memory, alpha_cat, mask
            )
            alpha_cumulative += alpha_tm1
            outputs.append(
                self.linear_projection(torch.cat([h_tm1[0], w_tm1], 1))
            )

        return torch.stack(outputs, -1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(dilation),
            nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.InstanceNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.InstanceNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self, vocab_size, enc_emb_size, enc_hidden_size, enc_n_layers,
        n_spkrs, spkr_size, rnn_size, align_size,
        att_size, att_n_filters, att_kernel_size,
        output_nc, ngf, n_downsampling, n_blocks
    ):
        super().__init__()

        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(align_size, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(True)
        ]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose1d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=4, stride=2, padding=1
                ),
                nn.InstanceNorm1d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(ngf, output_nc, kernel_size=7, padding=0)
        ]
        self.model = nn.Sequential(*model)
        self.encoder = RNNEncoder(vocab_size, enc_emb_size, enc_hidden_size // 2, enc_n_layers)
        self.aligner = Aligner(
            n_spkrs, spkr_size, enc_hidden_size, align_size, rnn_size,
            att_size, att_n_filters, att_kernel_size
        )

    def forward(self, T, spkrs, chars, chars_mask):
        memory = self.encoder(chars, chars_mask)
        enc = self.aligner(T, spkrs, memory, chars_mask)
        return self.model(enc)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.Conv1d(input_nc, ndf, kernel_size=15),
            nn.InstanceNorm1d(ndf),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)
            model["layer_%d" % n] = nn.Sequential(
                nn.Conv1d(
                    nf_prev, nf,
                    kernel_size=stride * 2,
                    stride=stride,
                    padding=stride // 2
                ),
                nn.InstanceNorm1d(nf),
                nn.LeakyReLU(0.2, True)
            )

        nf_prev = nf
        nf = min(nf * 2, 1024)

        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            nn.Conv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(nf),
            nn.LeakyReLU(0.2, True)
        )

        model["layer_%d" % (n_layers + 2)] = nn.Conv1d(nf, 1, kernel_size=5, stride=1, padding=2)

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, input_nc, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model["disc_%d" % i] = NLayerDiscriminator(
                input_nc, ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(
            3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


if __name__ == '__main__':
    vocab_size = 60
    emb_size = 128
    enc_hidden_size = 256
    enc_n_layers = 3
    align_size = 512
    dec_hidden_size = 512
    att_size = 256
    att_n_filters = 32
    att_kernel_size = 11
    output_nc = 3
    ngf = 64
    n_blocks = 5
    n_downsampling = 4
    T = 512

    chars = torch.randint(0, 60, (1, 24)).long().cuda()
    chars_mask = torch.ones_like(chars).byte().cuda()

    netG = Generator(
        vocab_size, emb_size, enc_hidden_size, enc_n_layers,
        dec_hidden_size, align_size, att_size, att_n_filters, att_kernel_size,
        output_nc, ngf, n_downsampling, n_blocks
    ).cuda()
    print(netG(T, chars, chars_mask))

