"""Message encoder for four deepest 128x128 Haar subbands.

For a 4 KB payload (32768 bits), each subband receives 8192 bits arranged as
64x128. The feature map is resized to 128x128 before coupling, so the image
is still decomposed to 128x128 and the network does not process 1024x1024
feature maps internally.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super().__init__()
        conv = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)
        if init_zero:
            nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)
            self.layers = conv
        else:
            self.layers = nn.Sequential(conv, nn.LeakyReLU(inplace=True))
    def forward(self, x): return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, H=128, message_length=32768, channels=64):
        super().__init__()
        if message_length % 4 != 0: raise ValueError("message_length must be divisible by 4")
        self.message_length = message_length
        self.part_length = message_length // 4
        self.part_h = self.part_length // H
        if self.part_h * H != self.part_length:
            raise ValueError("Each subband payload must be representable as part_h x H")
        self.part_w = H
        if self.part_h > H: raise ValueError("per-subband payload is too large for H")
        self.net = nn.Sequential(ConvRelu(1, channels), ConvRelu(channels, channels), ConvRelu(channels, 3, init_zero=True))

    def forward(self, message):
        if message.ndim != 2 or message.shape[1] != self.message_length:
            raise ValueError(f"expected (B,{self.message_length}) message")
        b = message.shape[0]; l = self.part_length
        out = []
        for i in range(4):
            part = message[:, i*l:(i+1)*l].reshape(b, 1, self.part_h, self.part_w)
            part = F.interpolate(part, size=(self.part_w, self.part_w), mode="bilinear", align_corners=False)
            out.append(self.net(part))
        return tuple(out)



