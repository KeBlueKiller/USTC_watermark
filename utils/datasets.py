import glob
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import config as c
try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

def to_rgb(image):
    if image.mode == 'RGB': return image
    return image.convert('RGB')

class ImageDataset(Dataset):
    def __init__(self, root, ext, transform):
        self.files = natsorted(glob.glob(root.rstrip('/\\') + '/*.' + ext))
        self.transform = transform
        if not self.files: raise FileNotFoundError(f'No *.{ext} images found in {root}')
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with Image.open(self.files[index]) as im:
            return self.transform(to_rgb(im))

def build_loaders():
    transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.Resize((c.cropsize,c.cropsize)), T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
    val_transform = T.Compose([T.Resize((c.cropsize_val,c.cropsize_val)), T.ToTensor(), T.Normalize([.5]*3,[.5]*3)])
    train_ds = ImageDataset(c.TRAIN_PATH, c.format_train, transform)
    val_ds = ImageDataset(c.VAL_PATH, c.format_val, val_transform)
    return (DataLoader(train_ds,batch_size=c.batch_size,shuffle=True,num_workers=c.num_workers,pin_memory=True,drop_last=True),
            DataLoader(val_ds,batch_size=c.batchsize_val,shuffle=False,num_workers=max(0,c.num_workers//2),pin_memory=True,drop_last=False))

class Test_Dataset(ImageDataset):
    def __init__(self,path,format):
        super().__init__(path,format,T.Compose([T.Resize((c.cropsize,c.cropsize)),T.ToTensor(),T.Normalize([.5]*3,[.5]*3)]))
