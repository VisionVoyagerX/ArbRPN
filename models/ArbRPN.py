#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ArbRPN.py
@Contact :   lihuichen@126.com
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
xxxxxxxxxx        LihuiChen      1.0         None
'''

# import lib

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import functools


class ResBlock(nn.Module):
    def __init__(self, inFe, outFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, outFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outFe, outFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class ArbRPN(nn.Module):
    def __init__(self, opt=None, **kwargs):
        super(ArbRPN, self).__init__()
        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std = kwargs.get('mslr_std')
        self.pan_mean = kwargs.get('pan_mean')
        self.pan_std = kwargs.get('pan_std')

        hid_dim = 64
        input_dim = 64
        num_resblock = 3
        self.num_cycle = 5

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)

        self.hidden_unit_forward_list = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()

        for _ in range(self.num_cycle):
            compress_1 = (nn.Conv2d(hid_dim + input_dim +
                          hid_dim, hid_dim, 1, 1, 0))
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(
                nn.Sequential(compress_1, resblock_1))

            compress_2 = (nn.Conv2d(hid_dim + input_dim +
                          hid_dim, hid_dim, 1, 1, 0))
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(
                nn.Sequential(compress_2, resblock_2))

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)

        self.apply(self.init_weights)

        ####################
        # initialize
        ####################

    def weights_init_normal(self, m, std=0.02):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                print('initializing [%s] ...' % classname)
                init.normal_(m.weight.data, 0.0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, std)
            init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(self, m, scale=1):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                print('initializing [%s] ...' % classname)
                init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.kaiming_normal_(m.weight.data, a=0,
                                 mode='fan_in', nonlinearity='relu')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.constant_(m.weight.data, 1.0)
            m.weight.data *= scale
            init.constant_(m.bias.data, 0.0)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if classname != "MeanShift":
                print('initializing [%s] ...' % classname)
                init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, (nn.Linear)):
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def init_weights(self, net, init_type='kaiming', scale=1, std=0.02):
        # scale for 'kaiming', std for 'normal'.
        print('initialization method [%s]' % init_type)
        if init_type == 'normal':
            weights_init_normal_ = functools.partial(
                self.weights_init_normal, std=std)
            net.apply(weights_init_normal_)
        elif init_type == 'kaiming':
            weights_init_kaiming_ = functools.partial(
                self.weights_init_kaiming, scale=scale)
            net.apply(weights_init_kaiming_)
        elif init_type == 'orthogonal':
            net.apply(self.weights_init_orthogonal)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)

    '''def __init__weights(self, m):
        init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()'''

    def forward(self, pan, ms, mask=None, is_cat_out=False):
        '''
        :param ms: LR ms images
        :param pan: pan images
        :param mask: mask to record the batch size of each band
        :return:
            HR_ms: a list of HR ms images,
        '''

        if mask is None:
            mask = [1 for _ in range(ms.shape[1])]
            is_cat_out = True

        ms = ms.split(1, dim=1)
        pan_state = self.wrapper(pan)
        hidden_state = pan_state
        blur_ms_list = []

        backward_hidden = []
        for idx, band in enumerate(ms):
            band = F.interpolate(
                band[:mask[idx]], scale_factor=4, mode='bicubic', align_corners=False)
            blur_ms_list.append(band)
            backward_hidden.append(self.conv1(band))

        backward_hidden = backward_hidden[::-1]
        for idx_cycle in range(self.num_cycle):
            # forward recurrence
            forward_hidden = []
            for idx in range(len(blur_ms_list)):
                hidden_state = hidden_state[:mask[idx]]
                band = torch.cat(
                    (backward_hidden[-(idx+1)], hidden_state, pan_state[:mask[idx]]), dim=1)
                hidden_state = self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)
            # backward recurrence
            backward_hidden = []
            for idx in range(len(blur_ms_list)):
                start_pan_stat = hidden_state.shape[0]
                hidden_state = torch.cat(
                    (hidden_state, pan_state[start_pan_stat:mask[-(idx+1)]]), dim=0)
                band = torch.cat(
                    (forward_hidden[-(idx + 1)], hidden_state, pan_state[:mask[-(idx+1)]]), dim=1)
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        HR_ms = []
        for idx in range(len(blur_ms_list)):
            band = self.conv2(backward_hidden[-(idx+1)])
            band = band + blur_ms_list[idx]
            HR_ms.append(band)
        return torch.cat(HR_ms, dim=1)


class myloss(nn.Module):
    def __init__(self, opt=None):
        super(myloss, self).__init__()

    def forward(self, ms, HR, mask=None):
        diff = 0
        count = 0
        HR = torch.split(HR, 1, dim=1)
        if mask is None:
            mask = [1 for _ in range(len(HR))]
            ms = ms[0]
        for idx, (band, hr) in enumerate(zip(ms, HR)):
            b, t, h, w = band.shape
            count += b * t * h * w
            diff += torch.sum(torch.abs(band - hr[:mask[idx]]))
        return diff / count


if __name__ == "__main__":
    model = ArbRPN()
    lr = torch.randn(1, 4, 16, 16)
    pan = torch.randn(1, 1, 64, 64)
    result = model(pan, lr)
    print(result.shape)
    print(1)
