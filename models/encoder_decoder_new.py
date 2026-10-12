"""
FED (Flow-based Encoder/Decoder) and INL (Invertible Noise Layer).

FED embeds message into all 4 DWT subbands (LL, LH, HL, HH) via INN.

Forward (encoding):
    cover_img -> Haar cascade -> (LL, LH, HL, HH) + intermediates
    INN forward: [(LL,LH,HL,HH), message] -> [(LL',LH',HL',HH'), left_noise]
    Haar inverse: (LL',LH',HL',HH') + intermediates -> stego_img
    Return: stego_img, left_noise

Reverse (decoding):
    stego_img -> Haar cascade -> (LL, LH, HL, HH) + intermediates
    INN reverse: [(LL,LH,HL,HH), z_noise] -> [(LL_rec,LH_rec,HL_rec,HH_rec), rec_msg]
    Haar inverse: rec subbands + intermediates -> rec_img
    Return: rec_img, rec_message
"""
import torch.nn as nn
from models.Inn_new import Noise_INN, INN
from block.Haar_new import HaarPyramid


class INL(nn.Module):
    """Invertible Noise Layer for black-box distortion simulation."""
    def __init__(self):
        super(INL, self).__init__()
        self.model = Noise_INN()
        self.haar = HaarPyramid(channel_in=3, levels=1)

    def forward(self, x, rev=False):
        if not rev:
            (LL, LH, HL, HH), intermediates = self.haar(x, rev=False)
            # Noise_INN expects [low_freq, high_freq_concat]
            # For single-level: LL is (B,3,H/2,W/2), need to concat LH,HL,HH -> (B,9,H/2,W/2)
            import torch
            high = torch.cat([LH, HL, HH], dim=1)
            out = self.model([LL, high])
            out_LL, out_high = out[0], out[1]
            # Split high back
            C = LL.shape[1]
            out_LH = out_high[:, 0:C, :, :]
            out_HL = out_high[:, C:2*C, :, :]
            out_HH = out_high[:, 2*C:3*C, :, :]
            rec = self.haar(((out_LL, out_LH, out_HL, out_HH), intermediates), rev=True)
            return rec
        else:
            (LL, LH, HL, HH), intermediates = self.haar(x, rev=False)
            import torch
            high = torch.cat([LH, HL, HH], dim=1)
            out = self.model([LL, high], rev=True)
            out_LL, out_high = out[0], out[1]
            C = LL.shape[1]
            out_LH = out_high[:, 0:C, :, :]
            out_HL = out_high[:, C:2*C, :, :]
            out_HH = out_high[:, 2*C:3*C, :, :]
            rec = self.haar(((out_LL, out_LH, out_HL, out_HH), intermediates), rev=True)
            return rec


class FED(nn.Module):
    """
    Flow-based Encoder/Decoder.

    Embeds message into all 4 DWT subbands via INN.
    """
    def __init__(self, diff=False, message_length=32768, cropsize=1024, levels=3, channels=64):
        super(FED, self).__init__()
        self.haar = HaarPyramid(channel_in=3, levels=levels)
        subband_H = cropsize // (2 ** levels)
        self.model = INN(
            H=subband_H,
            message_length=message_length,
            channels=channels,
            diff=diff
        )

    def forward(self, x, rev=False, message=None):
        """
        Forward:
            x: cover_img (B,3,H,W) or [cover_img, message]
            Returns: (stego_img, left_noise)
        Reverse:
            x: [stego_img, z_noise]
            Returns: (rec_img, rec_message)
        """
        if not rev:
            # Parse input
            if isinstance(x, (list, tuple)):
                cover_img, message = x[0], x[1]
            else:
                cover_img = x
            assert message is not None, "Forward needs message (B, message_length)"

            # Haar decompose
            (LL, LH, HL, HH), intermediates = self.haar(cover_img, rev=False)

            # INN forward: embed message into all 4 subbands
            subbands = (LL, LH, HL, HH)
            stego_subbands, left_noise = self.model([subbands, message], rev=False)

            # Haar reconstruct
            stego_img = self.haar((stego_subbands, intermediates), rev=True)

            return stego_img, left_noise

        else:
            # Parse input
            if isinstance(x, (list, tuple)):
                stego_img, z_noise = x[0], x[1]
            else:
                raise ValueError("Reverse requires [stego_img, z_noise]")

            # Haar decompose
            (LL, LH, HL, HH), intermediates = self.haar(stego_img, rev=False)

            # INN reverse: extract message from all 4 subbands
            subbands = (LL, LH, HL, HH)
            rec_subbands, rec_message = self.model([subbands, z_noise], rev=True)

            # Haar reconstruct
            rec_img = self.haar((rec_subbands, intermediates), rev=True)

            return rec_img, rec_message

