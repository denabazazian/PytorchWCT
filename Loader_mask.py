from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,content_mask_path, style_mask_path, fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.content_mask_path = content_mask_path
        self.style_mask_path = style_mask_path
        self.fineSize = fineSize
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Scale(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentMaskImgPath = os.path.join(self.content_mask_path,self.image_list[index])
        styleMaskImgPath = os.path.join(self.style_mask_path,self.image_list[index])

        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)
        contentMaskImg = default_loader(contentMaskImgPath)
        styleMaskImg = default_loader(styleMaskImgPath)
        # resize
        if(self.fineSize != 0):
            w,h = contentImg.size
            #pdb.set_trace()
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
                    contentMaskImg = contentMaskImg.resize((neww,newh))
                    styleMaskImg = styleMaskImg.resize((neww,newh))


            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
                    contentMaskImg = contentMaskImg.resize((neww,newh))
                    styleMaskImg = styleMaskImg.resize((neww,newh))



        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        contentMaskImg = transforms.ToTensor()(contentMaskImg)
        styleMaskImg = transforms.ToTensor()(styleMaskImg)

        return contentImg.squeeze(0),styleImg.squeeze(0),contentMaskImg.squeeze(0),styleMaskImg.squeeze(0),self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
