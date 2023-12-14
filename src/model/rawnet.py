import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=0, min_band_hz=0, # hint 6
                 filter_type="s1"):

        super().__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        if filter_type == "s1":
            mel = np.linspace(self.to_mel(low_hz),
                            self.to_mel(high_hz),
                            self.out_channels + 1)
        else: # s2, fixed inverse Mel-scaled
            mel = np.linspace(self.to_mel(high_hz),
                            self.to_mel(low_hz),
                            self.out_channels + 1)
        hz = self.to_hz(mel)


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))) # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)


        # (1, kernel_size / 2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes




    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left, dims=[1])


        band_pass=torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)


        band_pass = band_pass / (2 * band[:, None])


        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
    


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation2 = nn.LeakyReLU(negative_slope=0.3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pooling = nn.MaxPool1d(kernel_size=3)
        self.fms = nn.Linear(out_channels, out_channels) # wow so-called linear ATTENTION


    def forward(self, x):
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.max_pooling(x)
        x = self.fms(x.view(x.shape[0], -1, x.shape[1]))
        x = x.view(x.shape[0], -1, x.shape[1])
        return x
    
class RawNet2(nn.Module):
    def __init__(self, 
                 use_abs_after_sinc: bool = True,
                 use_bn_before_gru: bool = True,
                 filters: list = [20, 128], # hint 3
                 sinc_out: int = 20,
                 sinc_filter: int = 1024, # hint 3
                 sinc_filter_type: str = "s1",
                 n_res_blocks_first: int = 2,
                 n_res_blocks_second: int = 4,
                 gru_hidden: int = 1024,
                 gru_layers: int = 3) -> None:
        super().__init__()
        self.use_abs_after_sinc = use_abs_after_sinc
        self.use_bn_before_gru = use_bn_before_gru
        self.sinc = SincConv(sinc_out, sinc_filter, filter_type=sinc_filter_type)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.bn_sinc = nn.BatchNorm1d(sinc_out)

        resblocks_first = [ResBlock(sinc_out, filters[0])]
        resblocks_first.extend([ResBlock(filters[0], filters[0]) for _ in range(n_res_blocks_first - 1)])
        self.resblocks_first = nn.Sequential(*resblocks_first)

        resblocks_second = [ResBlock(filters[0], filters[1])]
        resblocks_second.extend([ResBlock(filters[1], filters[1]) for _ in range(n_res_blocks_second - 1)])
        self.resblocks_second = nn.Sequential(*resblocks_second)

        self.bn = nn.BatchNorm1d(filters[1])
        self.GRU = nn.GRU(input_size=filters[1], hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 2)


    def forward(self, x):
        x = self.sinc(x)
        x = self.pooling(x)
        x = self.bn_sinc(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        # hint 1: You have to take the absolute value of sinc-layer output.
        if self.use_abs_after_sinc:
            x = torch.abs(x)
        for block in self.resblocks_first:
            x = block(x)
        for block in self.resblocks_second:
            x = block(x)
        # hint 2: Use 3-layer GRU, do BN and LeakyReLU before.
        if self.use_bn_before_gru:
            x = self.bn(x)
            x = F.leaky_relu(x, negative_slope=0.3)
        x, _ = self.GRU(x.view(x.shape[0], -1, x.shape[1]))
        x = self.fc(x[:, -1, :])
        # x = F.softmax(x, dim=1)
        return {
            "bonafide_scores": x[:, 1],
            "other_scores": x[:, 0],
            "logits": x
        }

