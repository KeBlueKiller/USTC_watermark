import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .jpeg import Identity
from .middle_filter import Jpeg, JpegTest, JpegSS, JpegMask

class Combined(nn.Module):
    def __init__(self, layers=None): super().__init__(); self.layers=nn.ModuleList(layers or [Identity()])
    def forward(self,image_and_cover): return self.layers[random.randrange(len(self.layers))](image_and_cover)

class GN(nn.Module):
    def __init__(self,var,mean=0): super().__init__(); self.var=var; self.mean=mean
    def forward(self,pair):
        image,_=pair; return image+torch.randn_like(image)*(self.var**0.5)+self.mean

class GF(nn.Module):
    def __init__(self,sigma,kernel=7): super().__init__(); self.sigma=sigma; self.kernel=kernel
    def forward(self,pair):
        image,_=pair; k=self.kernel; ax=torch.arange(k,device=image.device,dtype=image.dtype)-(k-1)/2; g=torch.exp(-(ax**2)/(2*self.sigma**2)); g=g/g.sum(); ker=(g[:,None]*g[None,:]).expand(image.size(1),1,k,k); return F.conv2d(F.pad(image,(k//2,)*4,mode='reflect'),ker,groups=image.size(1))

class MF(nn.Module):
    def __init__(self,kernel): super().__init__(); self.kernel=kernel
    def forward(self,pair):
        image,_=pair; k=self.kernel; x=F.pad(image,(k//2,)*4,mode='reflect').unfold(2,k,1).unfold(3,k,1); return x.contiguous().view(*x.shape[:4],-1).median(-1).values

class SP(nn.Module):
    def __init__(self,prob): super().__init__(); self.prob=prob
    def forward(self,pair):
        image,_=pair; r=torch.rand_like(image); return torch.where(r<self.prob/2,torch.zeros_like(image),torch.where(r<self.prob,torch.ones_like(image),image))

class Dropout(nn.Module):
    def __init__(self,prob): super().__init__(); self.prob=prob
    def forward(self,pair):
        image,cover=pair; return torch.where(torch.rand_like(image)>self.prob,image,cover)

class Cropout(nn.Module):
    def __init__(self,height_ratio,width_ratio): super().__init__(); self.hr=height_ratio; self.wr=width_ratio
    def forward(self,pair):
        image,cover=pair; h,w=image.shape[-2:]; hh=max(1,int(h*self.hr)); ww=max(1,int(w*self.wr)); hs=random.randint(0,h-hh); ws=random.randint(0,w-ww); out=cover.clone(); out[:,:,hs:hs+hh,ws:ws+ww]=image[:,:,hs:hs+hh,ws:ws+ww]; return out

class Crop(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, pair):
        image, _ = pair
        height, width = image.shape[-2:]
        crop_height = max(1, int(height * self.p ** 0.5))
        crop_width = max(1, int(width * self.p ** 0.5))
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        mask = torch.ones_like(image)
        mask[:, :, top:top + crop_height, left:left + crop_width] = 0
        return image * mask

class Scaling(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, pair):
        image, _ = pair
        height, width = image.shape[-2:]
        resized = F.interpolate(
            image,
            size=(max(1, int(height * self.p)), max(1, int(width * self.p))),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(
            resized,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )