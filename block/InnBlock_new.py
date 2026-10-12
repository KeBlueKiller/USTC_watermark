"""Invertible FIN coupling blocks, retaining the original four-subband design."""
import torch
import torch.nn as nn
from block.DenseNet import ResidualDenseBlock_out as DB
from block.Encoder_new import Encoder
from block.Decoder_new import Decoder

class Noise_INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__(); self.clamp=clamp; self.r=DB(input=3,output=9); self.y=DB(input=3,output=9); self.f=DB(input=9,output=3)
    def scale(self,s): return torch.exp(self.clamp * (2*torch.sigmoid(s)-1))
    def forward(self,x,rev=False):
        x1,x2=x
        if not rev:
            y1=x1+self.f(x2); return [y1,self.scale(self.r(y1))*x2+self.y(y1)]
        y2=(x2-self.y(x1))/self.scale(self.r(x1)); return [x1-self.f(y2),y2]

class INN_block(nn.Module):
    def __init__(self,H=128,message_length=32768,channels=64,clamp=2.0):
        super().__init__(); self.clamp=clamp; self.f=Encoder(H,message_length,channels); self.r=Decoder(H,message_length,channels); self.y=Decoder(H,message_length,channels)
    def scale(self,s): return torch.exp(self.clamp * (2*torch.sigmoid(s)-1))
    def forward(self,x,rev=False):
        x1,x2=x
        if not rev:
            t2=self.f(x2); y1=tuple(x1[i]+t2[i] for i in range(4)); return [y1,self.scale(self.r(*y1))*x2+self.y(*y1)]
        s=self.r(*x1); y2=(x2-self.y(*x1))/self.scale(s); t2=self.f(y2); return [tuple(x1[i]-t2[i] for i in range(4)),y2]
