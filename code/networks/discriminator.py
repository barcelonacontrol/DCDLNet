#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 下午9:34
# @Author  : chuyu zhang
# @File    : discriminator.py
# @Software: PyCharm

import torch.nn as nn
import torch.nn.functional as F
import torch


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Linear(ndf*8, 2)
        self.avgpool = nn.AvgPool2d((7, 7))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


class FC3DDiscriminator_first(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminator_first, self).__init__()
            # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(ndf * 8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def classify_(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)
        return x
    def forward(self, map1, map2, image):
        logits1 = self.classify_(map1, image)
        logits2 = self.classify_(map2, image)
        out = torch.cat([logits1, logits2], dim=0)
        return out
class FC3DDiscriminator_new2(nn.Module):

    def __init__(self, num_classes, ndf=32, n_channel=1):
        super(FC3DDiscriminator_new2, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv01 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(2 * ndf, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map1, map2, image):

        map_feature = self.conv0(map1)
        map_feature2 = self.conv01(map2)
        image_feature = self.conv1(image)
        x = torch.add(image_feature, map_feature)
        x1 = torch.add(image_feature, map_feature2)
        x = torch.cat([x1, x], dim=1)
        batch_size = 2 * x.shape[0]
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class FC3DDiscriminator_new(nn.Module):

    def __init__(self, num_classes, ndf=32, n_channel=1):
        super(FC3DDiscriminator_new, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv01 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*3, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf*3, ndf*6, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf*6, ndf*12, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(ndf*12, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map1, map2, image):

        map_feature = self.conv0(map1)
        map_feature2 = self.conv01(map2)
        image_feature = self.conv1(image)
        x = torch.add(image_feature, map_feature)
        x1 = torch.add(image_feature, map_feature2)
        x = torch.cat([x1, x], dim=0)
        batch_size = x.shape[0]
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class sSE_Module_3D(nn.Module):
    def __init__(self, channel):
        super(sSE_Module_3D, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv3d(in_channels=channel, out_channels=channel//16, kernel_size=1,stride=1,padding=0),
                nn.Conv3d(in_channels=channel//16, out_channels=1, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)
class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)
class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_planes
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, in_planes, in_planes//self.groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

        self.conv_1 = nn.Conv3d(in_planes,out_planes,1)
        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.in_planes, output.size(-3), output.size(-2), output.size(-1))
        output_f = self.conv_1(output)
        return output_f

class DYBAC_3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,stride, padding):
        super(DYBAC_3D, self).__init__()
        self.dybac = nn.Sequential(
           sSE_Module_3D(in_planes),
           Dynamic_conv3d(in_planes=in_planes, out_planes=out_planes,kernel_size=kernel_size,stride=stride,padding=padding),
           nn.BatchNorm3d(out_planes),
           nn.ReLU6(),
        )
    def forward(self, x):
        z = self.dybac(x)
        return z

class Discriminator_new2_DYBAC3D(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(Discriminator_new2_DYBAC3D, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv01 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(2 * ndf, 4 * ndf, kernel_size=4, stride=2, padding=1)

        self.conv3 = DYBAC_3D(4 * ndf, 8 * ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = DYBAC_3D(
            8 * ndf, 16 * ndf, kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool3d((6, 6, 6))  # (D/16, W/16, H/16)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(ndf * 8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        # self.Softmax = nn.Softmax()

    def classify_(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)

        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x

    def forward(self, map1, map2, image):
        map_feature = self.conv0(map1)
        map_feature2 = self.conv01(map2)
        image_feature = self.conv1(image)
        x = torch.add(image_feature, map_feature)
        x1 = torch.add(image_feature, map_feature2)
        x = torch.cat([x1, x], dim=1)
        batch_size = 2 * x.shape[0]
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class Discriminator_first_DYBAC3D(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(Discriminator_first_DYBAC3D, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(
            num_classes, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            n_channel, ndf, kernel_size=3, stride=2, padding=1)

        self.conv2 = DYBAC_3D(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = DYBAC_3D(
            ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = DYBAC_3D(
            ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool3d((6, 6, 6))  # (D/16, W/16, H/16)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        # self.Softmax = nn.Softmax()

    def classify_(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)

        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
    def forward(self, map1, map2, image):
        logits1 = self.classify_(map1, image)
        logits2 = self.classify_(map2, image)
        out = torch.cat([logits1, logits2], dim=0)
        return out
class Discriminator_DYBAC3D(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(Discriminator_DYBAC3D, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(
            num_classes, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            n_channel, ndf, kernel_size=3, stride=2, padding=1)

        self.conv2 = DYBAC_3D(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = DYBAC_3D(
            ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = DYBAC_3D(
            ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool3d((6, 6, 6))  # (D/16, W/16, H/16)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        # self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)

        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class FC3DDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminator, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((7, 7, 5))
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x
class FC3DDiscriminatorNIH_first(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminatorNIH_first, self).__init__()
            # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((13, 10, 9))
        self.classifier = nn.Linear(ndf * 8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def classify_(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)
        return x
    def forward(self, map1, map2, image):
        logits1 = self.classify_(map1, image)
        logits2 = self.classify_(map2, image)
        out = torch.cat([logits1, logits2], dim=0)
        return out

class FC3DDiscriminatorNIH(nn.Module):
    def __init__(self, num_classes, ndf=64, n_channel=1):
        super(FC3DDiscriminatorNIH, self).__init__()
        # downsample 16
        self.conv0 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d((13, 10, 9))
        self.classifier = nn.Linear(ndf*8, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.Softmax = nn.Softmax()

    def forward(self, map, image):
        batch_size = map.shape[0]
        map_feature = self.conv0(map)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.reshape((batch_size, 2))
        # x = self.Softmax(x)

        return x

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = FC3DDiscriminator(num_classes=1)
    input = torch.randn(4, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,input))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    model = FCDiscriminatorDAP(num_classes=2)
    input = torch.randn(4, 2, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    import ipdb; ipdb.set_trace()