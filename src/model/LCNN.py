import torch
from torch import nn
import torchaudio


# https://arxiv.org/abs/1511.02683
class MFM(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        first, second = torch.split(x, self.out_channels, dim=1) # channel_dim is second in our data
        return torch.max(first, second)
    
class Flattener(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_dim=1, end_dim=-1): # start dim is 1 because we keep batch_size
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)


class LCNN(nn.Module):
    def __init__(self, 
                 frontend=1,
                 dropout=0.1): # from Improved Lightcnn with Attention Modules for Asv Spoofing Detection,
                                # original paper mentions 0.75
        super().__init__()

        if frontend == 1:
            self.frontend = torchaudio.transforms.Spectrogram(
                n_fft=1724,
                win_length=1724,
                hop_length=int(0.0081 * 16000),
                window_fn=torch.blackman_window,
            ) # STFT
        else: 
            self.frontend = torchaudio.transforms.LFCC(
                speckwargs={
                    "n_fft": 1724,
                    "win_length": 1724,
                    "hop_length": int(0.0081 * 16000),
                    "window_fn": torch.blackman_window
                }
            )


        # for more info about magic numbers see original paper: https://arxiv.org/pdf/1904.05576.pdf
        # not sure about the number of in_channels
        # (B, C, H, W)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            MFM(out_channels=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            MFM(out_channels=32),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM(out_channels=48),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1), stride=(1, 1)),
            MFM(out_channels=48),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM(out_channels=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM(out_channels=64),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM(out_channels=32),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
            MFM(out_channels=32),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            MFM(out_channels=32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Flattener(),
            nn.Linear(in_features=53*37*32, out_features=160),
            MFM(out_channels=80),
            nn.BatchNorm1d(num_features=80),
            nn.Dropout(dropout),
            nn.Linear(in_features=80, out_features=2)
        )

        self._init_weights()


    '''
    LCNN weights were initialized using normal Kaiming initialization.
    '''
    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        spec = self.frontend(x)

        # Only the first 600 features for each file were used as LCNN input in all single systems
        spec = torch.nn.functional.pad(spec, (0, 600 - spec.shape[-1]))

        for layer in self.layers:
            spec = layer(spec)

        return {
            "bonafide_scores": spec[:, 0],
            "other_scores": spec[:, 1],
            "logits": spec
        }




        