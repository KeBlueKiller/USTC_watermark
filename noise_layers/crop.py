import random
import torch.nn as nn

class Combined(nn.Module):
    def __init__(self, layers=None):
        super().__init__(); self.layers=nn.ModuleList(layers or [nn.Identity()])
    def forward(self, image_and_cover):
        return self.layers[random.randint(0,len(self.layers)-1)](image_and_cover)
