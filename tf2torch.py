import torch
import os
import sys
sys.path.append('../cls')
from model_cls_pytorch import get_model
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import numpy as np
import csv

if __name__ == '__main__':
    save_dir = '../cls/ckpts/model'

    model_path = os.path.join(save_dir, 'ResVGG_GN_001.ckpt')
    # net, loss, optimizer= get_model('test', 'BCELoss')# test train

    torch_dict = torch.load(model_path)['state_dict']##.state_dict()
    # print('model layer1:', torch_dict['forw0.0.weight'].shape)# layer0conv_W,forw0.0.weight
    print('torch:',  torch_dict['forw1.0.normal1.gama'][0,:,0,0,0])
    # print('torch:',  torch_dict['forw0.0.bias'])

    # dict_name = list(torch_dict)
    # for i, k in enumerate(dict_name):
    #     print(k, torch_dict[k].shape)

    tfModelPath = '../cls/ckpts/model/resnet.pd-50000'
    model_reader = pywrap_tensorflow.NewCheckpointReader(tfModelPath)
    tf_dict = model_reader.get_variable_to_shape_map()

    # layer0 = model_reader.get_tensor('layer0conv_W')
    # layer0 = torch.tensor(np.transpose(layer0, (4, 3, 0, 1, 2)).astype(np.float32))
    # print('tf:',  layer0.shape)# var_dict['layer0conv_W']
    # print('tf:',  layer0[0,0,:,:,:])
    layer0 = model_reader.get_tensor('layer0groupgroup_gama')
    print('tf:',  layer0)
    # print('tf:',  torch.from_numpy(layer0.astype(np.float32)))

    # for key in tf_dict:
    #     print(key, model_reader.get_tensor(key).shape)

    # MappingPath = '../cls/ckpts/nameMapping.csv'
    # modelList = []
    # with open(MappingPath, "r") as file:
    #     fileContext = csv.reader(file)  # classify_metrics
    #     for line in fileContext:
    #         modelList.append(line)
    # modelList = np.array(modelList[1:])[:, :]#.tolist()
    #
    # # print('list:', modelList.shape)
    # for tfModelName, torchModelName in modelList:
    #     tf_key = model_reader.get_tensor(tfModelName)
    #     if len(torch_dict[torchModelName].shape) > 2:# need > 2(the shape of fc layer)
    #         if 'weight' in torchModelName:
    #             # print('tf:',tfModelName)
    #             # weight layouts  |     tensorflow     |        pytorch      |    transpose     |
    #             # conv2d_transpose (H, W, C, batch) ->   (batch, C, H, W)      (3, 2, 0, 1)
    #             # conv3d_transpose (batch, D, H, W, C) -> (batch, C, D, H, W) (4, 3, 0, 1, 2)
    #             torch_dict[torchModelName] = torch.tensor(np.transpose(tf_key, (4, 3, 0, 1, 2)).astype(np.float32))
    #         else:
    #             torch_dict[torchModelName] = torch.tensor(tf_key.astype(np.float32)).view(1, -1, 1, 1, 1)
    #     else:
    #         if 'fcBlock' in torchModelName and 'weight' in torchModelName:
    #             # print('size:', tf_key.shape)
    #             torch_dict[torchModelName] = torch.tensor(np.transpose(tf_key, (1, 0)).astype(np.float32))
    #         else:
    #             torch_dict[torchModelName] = torch.tensor(tf_key.astype(np.float32))
    #
    # # module_dict = OrderedDict()
    # # for torchModelName, torchValue in torch_dict.items():
    # #     module_dict['module.' + torchModelName] = torchValue#torch_dict[torchModelName]
    # save_dir = '../cls/ckpts/model'
    # torch.save({
    #     'epoch': 0,
    #     'save_dir': save_dir,
    #     'state_dict': torch_dict,
    #     'cfgs': ''},
    #     os.path.join(save_dir, 'ResVGG_GN_001.ckpt'))
    # print('save finish!')

