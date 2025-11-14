import math
import torch.nn as nn
import torch.nn.functional as F
import torch


# TODO: implement dot_product and other non-local formats
class ModulatedAttLayer(nn.Module):

    def __init__(self, in_channels, reduction=2, mode='embedded_gaussian'):
        super(ModulatedAttLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.conv_mask = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_spatial = nn.Linear(7 * 7 * self.in_channels, 7 * 7)
        # self.fc_channel = nn.Linear(7 * 7 * self.in_channels, self.in_channels)
        # self.fc_selector = nn.Linear(7 * 7 * self.in_channels, 1)

        # self.triplet_loss = TripletLoss(margin=0.2)

        self.init_weights()

    def init_weights(self):
        msra_list = [self.g, self.theta, self.phi]
        for m in msra_list:
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()
        self.conv_mask.weight.data.zero_()

    def embedded_gaussian(self, x):
        # embedded_gaussian cal self-attention, which may not strong enough
        batch_size = x.size(0)

        g_x = self.g(x.clone()).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x.clone()).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x.clone()).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)

        x_flatten = x.view(-1, 7 * 7 * self.in_channels)

        spatial_att = self.fc_spatial(x_flatten)
        spatial_att = spatial_att.softmax(dim=1)

        spatial_att = spatial_att.view(-1, 7, 7).unsqueeze(1)
        spatial_att = spatial_att.expand(-1, self.in_channels, -1, -1)

        final = spatial_att * mask + x

        return final, [x, spatial_att, mask]

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output, feature_maps = self.embedded_gaussian(x)
        else:
            raise NotImplemented("The code has not been implemented.")
        return output, feature_maps



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_modulatedatt=False, use_fc=False, dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_fc = use_fc
        self.use_dropout = True if dropout else False

        if self.use_fc:
            print('Using fc.')
            self.fc_add = nn.Linear(512 * block.expansion, 512)

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            print('Using self attention.')
            self.modulatedatt = ModulatedAttLayer(in_channels=512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer1':
            out = kwargs['coef']*out + (1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer2(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer2':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer3(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer3':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]
        out = self.layer4(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer4':
            out = kwargs['coef']*out+(1-kwargs['coef'])*out[kwargs['index']]

        return out


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model

def res10(cfg, use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, test=False,
          pretrain=False,
          pretrained_model =  None,
          last_layer_stride = None,
          flag=None):
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)
    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            resnet10 = init_weights(model=resnet10,
                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet10

def res152(cfg, use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=True,
                 test=False, pretrain=True, pretrained_model=None, last_layer_stride=2):
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)

    if not test:

        assert (caffe != stage1_weights)

        if caffe:
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='/path/to/checkpoints/resnet152.pth',
                                     caffe=True)
        elif stage1_weights:
            assert (dataset)
            print('Loading %s Stage 1 ResNet 152 Weights.' % dataset)
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet152
