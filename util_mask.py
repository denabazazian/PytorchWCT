from __future__ import division
import torch
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
from modelsNIPS import decoder1,decoder2,decoder3,decoder4,decoder5
from modelsNIPS import encoder1,encoder2,encoder3,encoder4,encoder5
import torch.nn as nn
import torch.nn.functional as F
import pdb

class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)
        vgg2 = load_lua(args.vgg2)
        decoder2_torch = load_lua(args.decoder2)
        vgg3 = load_lua(args.vgg3)
        decoder3_torch = load_lua(args.decoder3)
        vgg4 = load_lua(args.vgg4)
        decoder4_torch = load_lua(args.decoder4)
        vgg5 = load_lua(args.vgg5)
        decoder5_torch = load_lua(args.decoder5)


        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self,cF,sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF,1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature


    def wct2(self,cF,sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF,1).float() # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        iden = torch.eye(cFSize[0]).cuda().float()   #.double()
        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + iden
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        #iden = torch.eye(sFSize[0]).cuda().double()
        try:
            styleConv = torch.mm(sF,sF.t()).div(max(1, sFSize[1]-1)) + iden
        except:
            import pdb; pdb.set_trace()
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        #c_d = (c_e[0:k_c]).pow(-0.5)
        c_d = (c_e).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        diag_matrix = torch.diag(s_d)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],diag_matrix),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)

        return targetFeature


    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        csF.data.resize_(ccsF.size()).copy_(ccsF)
        return csF

    def large_dff(self,a,b):
        if(a / b >= 100):
            return True
        if(b / a >= 100):
            return True
        return False
     
    def scale_dialate(self,seg,W,H):
        # TODO: dialate
        seg = seg.view(1, 1, seg.shape[0], seg.shape[1])
        seg = F.interpolate(seg, size = (H, W), mode = 'nearest')
        return seg.squeeze()

    #def feature_wct(self, cF, sF, cmasks, smasks, alpha):
    def transform_mask(self, cF, sF, cmasks, smasks, alpha):    
            color_code_number = 1

            C, W, H = cF.size(0),cF.size(1),cF.size(2)
            _, W1, H1 = sF.size(0),sF.size(1),sF.size(2)
            targetFeature = cF.view(C,-1).clone()
            for i in range(color_code_number):
                cmask = cmasks[i].clone().squeeze(0)
                smask = smasks[i].clone().squeeze(0)
                if(torch.sum(cmask) >= 10 and torch.sum(smask) >= 10 and not self.large_dff(torch.sum(cmask),torch.sum(smask))):
                    cmaskResized = self.scale_dialate(cmask,W,H)
                    if(torch.max(cmaskResized) <= 0):
                        print('cmaskResized has no 1, ignore')
                        continue
                    cmaskView = cmaskResized.view(-1)
                    #pdb.set_trace()
                    #fgcmask = (cmaskView == 1).nonzero().squeeze(1)
                    fgcmask = (cmaskView > 0 ).nonzero().squeeze(1)

                    fgcmask = fgcmask.cuda()
                    cFView = cF.view(C,-1).cuda()
                    cFFG = torch.index_select(cFView,1,fgcmask.long())

                    smaskResized = self.scale_dialate(smask,W1,H1)
                    if(torch.max(smaskResized) <= 0):
                        print('smaskResized has no 1, ignore')
                        continue
                    smaskView = smaskResized.view(-1)
                    #fgsmask = (smaskView == 1).nonzero().squeeze(1)
                    fgsmask = (smaskView > 1).nonzero().squeeze(1)
                    fgsmask = fgsmask.cuda()
                    sFView = sF.view(C,-1).cuda()
                    sFFG = torch.index_select(sFView,1,fgsmask.long())

                    targetFeatureFG = self.wct2(cFFG.float(),sFFG.float())
                    targetFeature.index_copy_(1,fgcmask.cpu(),targetFeatureFG.cpu())
                    del fgcmask
                    del fgsmask
            targetFeature = targetFeature.view_as(cF)
            ccsF = alpha * targetFeature + (1.0 - alpha) * cF
            ccsF = ccsF.float().unsqueeze(0)
            return ccsF













