import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader_mask import Dataset
from util_mask import *
import scipy.misc
from torch.utils.serialization import load_lua
import time

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--content_mask_Path',default='images/mask/content',help='path to train')
parser.add_argument('--style_mask_Path',default='images/mask/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

# Data loading code
dataset = Dataset(args.contentPath,args.stylePath,args.content_mask_Path,args.style_mask_Path,args.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)
def styleTransfer(contentImg,styleImg,contentMaskImg,styleMaskImg,imname,csF):

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    cmF5 = wct.e5(contentMaskImg)
    smF5 = wct.e5(styleMaskImg)


    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    cmF5 = cmF5.data.cpu().squeeze(0)
    smF5 = smF5.data.cpu().squeeze(0)

    csF5 = wct.transform_mask(cF5,sF5,cmF5,smF5,args.alpha)   #(cF5,sF5,csF,args.alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    cmF4 = wct.e4(contentMaskImg)
    smF4 = wct.e4(styleMaskImg)

    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    cmF4 = cmF4.data.cpu().squeeze(0)
    smF4 = smF4.data.cpu().squeeze(0)

    csF4 = wct.transform_mask(cF4,sF4,cmF4,smF4,args.alpha)   #(cF4,sF4,csF,args.alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    cmF3 = wct.e3(contentMaskImg)
    smF3 = wct.e3(styleMaskImg)

    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    cmF3 = cmF3.data.cpu().squeeze(0)
    smF3 = smF3.data.cpu().squeeze(0)

    csF3 = wct.transform_mask(cF3,sF3,cmF3,smF3,args.alpha)     #(cF3,sF3,csF,args.alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    cmF2 = wct.e2(contentMaskImg)
    smF2 = wct.e2(styleMaskImg)

    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    cmF2 = cmF2.data.cpu().squeeze(0)
    smF2 = smF2.data.cpu().squeeze(0)

    csF2 = wct.transform_mask(cF2,sF2,cmF2,smF2,args.alpha)   #(cF2,sF2,csF,args.alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    cmF1 = wct.e1(contentMaskImg)
    smF1 = wct.e1(styleMaskImg)

    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    cmF1 = cmF1.data.cpu().squeeze(0)
    smF1 = smF1.data.cpu().squeeze(0)

    csF1 = wct.transform_mask(cF1,sF1,cmF1,smF1,args.alpha)   #(cF1,sF1,csF,args.alpha)
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))
    return

avgTime = 0
cImg = torch.Tensor()
sImg = torch.Tensor()
cMImg = torch.Tensor()
sMImg = torch.Tensor()

csF = torch.Tensor()
csF = Variable(csF)
if(args.cuda):
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    cMImg = cMImg.cuda(args.gpu)
    sMImg = sMImg.cuda(args.gpu)

    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)
for i,(contentImg,styleImg,contentMaskImg,styleMaskImg,imname) in enumerate(loader):
    imname = imname[0]
    print('Transferring ' + imname)
    if (args.cuda):
        contentImg = contentImg.cuda(args.gpu)
        styleImg = styleImg.cuda(args.gpu)
        contentMaskImg = contentMaskImg.cuda(args.gpu)
        styleMaskImg = styleMaskImg.cuda(args.gpu)

    cImg = Variable(contentImg,volatile=True)
    sImg = Variable(styleImg,volatile=True)
    cMImg = Variable(contentMaskImg,volatile=True)
    sMImg = Variable(styleMaskImg,volatile=True)

    start_time = time.time()
    # WCT Style Transfer
    styleTransfer(cImg,sImg,cMImg,sMImg,imname,csF)
    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))
    avgTime += (end_time - start_time)

print('Processed %d images. Averaged time is %f' % ((i+1),avgTime/(i+1)))
