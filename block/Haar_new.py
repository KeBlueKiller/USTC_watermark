"""
Haar wavelet transform with multi-level cascade.

Forward: image -> multi-level DWT -> (LL, LH, HL, HH) at deepest level + intermediates
Reverse: (LL, LH, HL, HH) at deepest level + intermediates -> reconstructed image

At each level, only the LL subband is further decomposed (standard wavelet cascade).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        # 4 Haar 2x2 kernels: LL, LH, HL, HH
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights, requires_grad=False)

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2,
                           groups=self.channel_in) / 4.0
            out = out.reshape(x.shape[0], self.channel_in, 4,
                              x.shape[2] // 2, x.shape[3] // 2)
            out = torch.transpose(out, 1, 2)
            out = out.reshape(x.shape[0], self.channel_in * 4,
                              x.shape[2] // 2, x.shape[3] // 2)

            LL = out[:, 0:self.channel_in, :, :]
            LH = out[:, self.channel_in:2 * self.channel_in, :, :]
            HL = out[:, 2 * self.channel_in:3 * self.channel_in, :, :]
            HH = out[:, 3 * self.channel_in:4 * self.channel_in, :, :]
            return LL, LH, HL, HH

        else:
            LL, LH, HL, HH = x
            x_cat = torch.cat((LL, LH, HL, HH), dim=1)
            out = x_cat.reshape(x_cat.shape[0], 4, self.channel_in,
                                x_cat.shape[2], x_cat.shape[3])
            out = torch.transpose(out, 1, 2)
            out = out.reshape(x_cat.shape[0], self.channel_in * 4,
                              x_cat.shape[2], x_cat.shape[3])
            return F.conv_transpose2d(out, self.haar_weights, bias=None,
                                      stride=2, groups=self.channel_in)


class HaarPyramid(nn.Module):
    """
    Multi-level Haar pyramid.

    Forward:
        Input: image (B, C, H, W)
        Output: (LL, LH, HL, HH) at deepest level, plus intermediates list for reconstruction

    Reverse:
        Input: (LL, LH, HL, HH) at deepest level, intermediates list
        Output: reconstructed image (B, C, H, W)

    The intermediates store (LH, HL, HH) at each level (except deepest) for reconstruction.
    """

    def __init__(self, channel_in=3, levels=3):
        super(HaarPyramid, self).__init__()
        self.levels = levels
        self.channel_in = channel_in
        self.haar_layers = nn.ModuleList([
            HaarDownsampling(channel_in) for _ in range(levels)
        ])

    def forward(self, x, rev=False):
        if not rev:
            # x: (B, C, H, W) image
            intermediates = []  # stores (LH, HL, HH) for each level
            current = x
            for i in range(self.levels):
                LL, LH, HL, HH = self.haar_layers[i](current, rev=False)
                if i < self.levels - 1:
                    # save high-freq for reconstruction, continue with LL
                    intermediates.append((LH, HL, HH))
                    current = LL
                else:
                    # deepest level: return all 4 subbands
                    return (LL, LH, HL, HH), intermediates
        else:
            # x = ((LL, LH, HL, HH), intermediates)
            (LL, LH, HL, HH), intermediates = x

            # reconstruct deepest level
            current = self.haar_layers[self.levels - 1](
                (LL, LH, HL, HH), rev=True
            )

            # reconstruct from inner to outer
            for i in reversed(range(self.levels - 1)):
                saved_LH, saved_HL, saved_HH = intermediates[i]
                current = self.haar_layers[i](
                    (current, saved_LH, saved_HL, saved_HH), rev=True
                )

            return current
