"""Message decoder for four deepest 128x128 Haar subbands."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from block.Encoder_new import ConvRelu

class Decoder(nn.Module):
    def __init__(self, H=128, message_length=32768, channels=64):
        super().__init__()
        if message_length % 4 != 0: raise ValueError("message_length must be divisible by 4")
        self.message_length = message_length
        self.part_length = message_length // 4
        self.part_h = self.part_length // H
        if self.part_h * H != self.part_length:
            raise ValueError("Each subband payload must be representable as part_h x H")
        self.net = nn.Sequential(ConvRelu(3, channels), ConvRelu(channels, channels), ConvRelu(channels, 1, init_zero=True))

    def forward(self, LL, LH, HL, HH):
        parts = []
        for band in (LL, LH, HL, HH):
            x = self.net(band)
            x = F.adaptive_avg_pool2d(x, (self.part_h, 128)).flatten(1)
            parts.append(x)
        return torch.cat(parts, dim=1)



