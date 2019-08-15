# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.groupNormal import GroupNorm3d
import os
debug = False# True False


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, normal_method='batch', stride=1, setDorp=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        if normal_method == 'batch':
            self.normal1 = nn.BatchNorm3d(planes)# it should be group
        elif normal_method == 'group':
            self.normal1 = GroupNorm3d(planes, 2, affine=True, track_running_stats=False)
        # self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        if normal_method == 'batch':
            self.normal2 = nn.BatchNorm3d(planes)# it should be group
        elif normal_method == 'group':
            self.normal2 = GroupNorm3d(planes, 2, affine=True, track_running_stats=False)
        # self.bn2 = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace = True)
        self.drop = nn.Dropout3d(p= setDorp , inplace = False)

        if stride != 1 or planes != in_planes:
            if normal_method == 'batch':
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size = 1, stride = stride),
                    nn.BatchNorm3d(planes))
            elif normal_method == 'group':
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size = 1, stride = stride),
                    GroupNorm3d(planes, 2, affine=True, track_running_stats=False))
            # self.shortcut = nn.Sequential(
            #     nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride),
            #     nn.BatchNorm3d(planes))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.drop(self.relu(self.normal1(self.conv1(x))))
        out = self.drop(self.relu(self.normal2(self.conv2(out))))
        # out = self.drop(self.relu(self.bn1(self.conv1(x))))
        # out = self.drop(self.relu(self.bn2(self.conv2(out))))

        out = F.relu(out)
        out += residual
        return out


class Res_cls(nn.Module):
    def __init__(self, cfg, drop=0.3):
        super(Res_cls, self).__init__()
        # def _create_conv_net(X, image_z, image_width, image_height, image_channel, drop, phase, n_class=1):
        in_planes, out_planes, normal = cfg['in_planes'], cfg['out_planes'], cfg['normal_method']
        self.forw1 = self._make_layer(in_planes[0], out_planes[0], stride=1, drop =drop, normal_method= normal)
        self.forw2 = self._make_layer(in_planes[1], out_planes[1], stride=1, drop =drop, normal_method= normal)
        self.forw3 = self._make_layer(in_planes[2], out_planes[2], stride=1, drop =drop, normal_method= normal)
        self.forw4 = self._make_layer(in_planes[3], out_planes[3], stride=1, drop =drop, normal_method= normal)
        self.forw5 = self._make_layer(in_planes[4], out_planes[4], stride=1, drop =drop, normal_method= normal)

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices =True)

        self.fcBlock1 = nn.Sequential(
            nn.Linear(3 * 3 * 3 * out_planes[4], 512),#
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=drop, inplace=False))

        # self.fcBlock2 = nn.Conv3d(512, 2, kernel_size=3, padding=1, bias=True)
        self.fcBlock2 = nn.Linear(512, 2)


    def _make_layer(self, in_planes, out_planes, stride, drop, normal_method='batch'):
        layers = []
        layers.append(BasicBlock(in_planes, out_planes, normal_method, stride, drop))
        return nn.Sequential(*layers)


    def forward(self, x):
        if debug: print('x:', x.size())
        # Vnet model
        out1 = self.forw1(x)#16
        if debug: print('1:', out1.size())
        out1_pool,indices0 = self.maxpool1(out1)
        # if debug: print '0.5:', out_pool.size()
        out2 = self.forw2(out1_pool)#32
        if debug: print('2:', out2.size())
        out2_pool,indices1 = self.maxpool2(out2)
        out3 = self.forw3(out2_pool)#64
        if debug: print('3:', out3.size())
        #out2 = self.drop(out2)
        out3_pool,indices2 = self.maxpool3(out3)
        out4 = self.forw4(out3_pool)#128
        if debug: print('4:', out4.size())
        out4_pool,indices3 = self.maxpool4(out4)
        out5 = self.forw5(out4_pool)#256
        if debug: print('5:', out5.size())

        # layer6->FC1
        out6 = out5.view(-1, 3 * 3 * 3 * 256)# shape=(?, 512)
        if debug: print('6:', out6.size())
        out6 = self.fcBlock1(out6)
        if debug: print('7:', out6.size())
        # layer7->output
        output = self.fcBlock2(out6)
        if debug: print('8:', output.size())
        return output


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            # if isinstance(alpha, Variable):
            #     self.alpha = alpha
            # else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        # print('alpha:', self.alpha)
        # print(N, C)
        P = F.softmax(inputs, dim=1)
        # print(P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask, ids)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----batch_loss------')
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def ResVGG(phase):
    """
    Constructs a ResVGG model.
    Config:......
    """
    if phase== 'train':
        drop = 0.3
    else:
        drop = 0
    cfg = {
        'in_planes': (1, 16, 32, 64, 128),
        'out_planes': (16, 32, 64, 128, 256),
        'normal_method': 'group'# batch group
    }
    return Res_cls(cfg, drop)


def get_model(phase= 'train', loss_name ='BCELoss'):
    net = ResVGG(phase)
    # loss = nn.CrossEntropyLoss()
    # loss = nn.BCELoss()
    if loss_name == 'BCELoss':
        loss = nn.BCEWithLogitsLoss()
    else:
        loss = FocalLoss(class_num=2, alpha=torch.Tensor([[0.25], [0.25]]))
    optimizer = torch.optim.Adam(net.parameters())
    return net, loss, optimizer#config,


if __name__ == '__main__':
    # loss = FocalLoss(class_num=2, alpha=torch.Tensor([[0.25], [0.25]]))#, size_average=False)#
    #
    # # conf_mask = torch.FloatTensor([0.0, 1.0, 0.0, 1.0, 1.0])-1
    # # conf_data = torch.FloatTensor([-0.1, -0.9, 0.0, -0.2, -0.2])
    # # conf_mask = torch.FloatTensor([ 109.3503, -134.3343]).cuda()
    # # conf_data = torch.FloatTensor([ 1.,  0.]).cuda()
    # loss.cuda()
    # conf_mask = torch.from_numpy(np.array([[ 109.3503, -134.3343],
    #         [  -2.1892,    2.1910],
    #         [ 609.3666, -514.9020],
    #         [   2.1850,   -2.1894],
    #         [ 101.6602,  -83.4581],
    #         [ 191.2697, -147.3313],
    #         [  20.8770,  -20.9577],
    #         [ 126.9506, -182.9222]]).astype(np.float32)).cuda()
    # conf_data = torch.from_numpy(np.array([[0],
    #         [1],
    #         [0],
    #         [0],
    #         [0],
    #         [0],
    #         [0],
    #         [0]])).cuda()
    # print(loss(conf_mask, conf_data))
    # # loss(conf_mask, conf_data)
    net, loss, optimizer= get_model('test', 'BCELoss')# test train

    # # add gpu to net
    # net = net.cuda()
    # loss = loss.cuda()
    # net = nn.DataParallel(net)
    # save_dir = './ckpts/model'
    # torch.save({
    #     'epoch': 0,
    #     'save_dir': save_dir,
    #     'state_dict': net.module.state_dict(),
    #     'cfgs': ''},
    #     os.path.join(save_dir, 'ResVGG_GN.ckpt'))
    # print('save finish!')

    # print('net:', net)
    input = torch.rand(2, 1, 48, 48, 48)
    print('input:', input.shape)
    output = net(input)
    print('output:', output)
