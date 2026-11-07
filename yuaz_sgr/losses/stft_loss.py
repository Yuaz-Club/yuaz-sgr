# YUAZ SGR
"""losses/stft_loss"""
import torch
import torch.nn as nn

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 论文: FFT/window size = 2^(5+i), i=1..6
        self.fft_sizes = [64, 128, 256, 512, 1024, 2048]
        # 论文: hop length = 25% of window length
        self.hop_lengths = [s // 4 for s in self.fft_sizes]

        self.win_lengths = self.fft_sizes

    def forward(self, x, x_hat):
        total = 0
        for fft, hop, win in zip(self.fft_sizes, self.hop_lengths, self.win_lengths):
            window = torch.hann_window(win, device=x.device)

            x_spec = torch.stft(x.squeeze(1), fft, hop, win,
                                window, return_complex=True).abs()
            xh_spec = torch.stft(x_hat.squeeze(1), fft, hop, win,
                                 window, return_complex=True).abs()

            # 论文 Eq (12): |||S(x)|-|S(x̂)|||²₂
            sc_loss = ((x_spec - xh_spec) ** 2).mean()
            total = total + sc_loss

            # 论文 Eq (12): ||log|S(x)|-log|S(x̂)|||²₂
            log_x = torch.log(torch.clamp(x_spec, min=1e-4))
            log_xh = torch.log(torch.clamp(xh_spec, min=1e-4))
            log_loss = ((log_x - log_xh) ** 2).mean()
            total = total + log_loss

        return total
