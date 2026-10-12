from torch import nn
from block.InnBlock_new import Noise_INN_block, INN_block

class Noise_INN(nn.Module):
    def __init__(self): super().__init__(); self.blocks=nn.ModuleList([Noise_INN_block() for _ in range(8)])
    def forward(self,x,rev=False):
        out=x
        for block in (reversed(self.blocks) if rev else self.blocks): out=block(out,rev=rev)
        return out

class INN(nn.Module):
    def __init__(self,H=128,message_length=32768,channels=64,diff=False):
        super().__init__(); self.diff=diff; self.message_length=message_length
        if diff: raise ValueError("diff=True is not supported for the 4KB payload experiment")
        self.blocks=nn.ModuleList([INN_block(H,message_length,channels) for _ in range(8)])
    def forward(self,x,rev=False):
        out=x
        for block in (reversed(self.blocks) if rev else self.blocks): out=block(out,rev=rev)
        return out
