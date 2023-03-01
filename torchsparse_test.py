from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.nn as nn
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard
import random
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
import torch
import logging
import paddle.sparse as sparse
import paddle.incubate as pi
import time
import sys
import torch
from torchsparse import nn as spnn
from torchsparse import SparseTensor

paddle.set_default_dtype("float32")
torch.set_default_dtype(torch.float32)


def generate_data(config):
    values = []
    indices = []
    print(config['nnz'])

    for i in range(config['nnz']):
        value = []
        idx = []
        for j in range(config['in_channels']):
            value.append(random.uniform(-1, -0.0001) * random.choice([-1, 1]))
        values.append(value)

        idx.append(random.randrange(0, config['batch_size']))
        idx.append(random.randrange(0, config['x']))
        idx.append(random.randrange(0, config['y']))
        idx.append(random.randrange(0, config['z']))
        indices.append(idx)
    return values, indices

config = [
    # 0
    {
        'batch_size': 1,
        'x': 41,
        'y': 1600,
        'z': 1408,
        'in_channels': 4,
        'out_channels': 16,
        'kernel_size': (3, 2, 1),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 136000
    },
    # 1
    {
        'batch_size': 1,
        'x': 41,
        'y': 1600,
        'z': 1408,
        'in_channels': 16,
        'out_channels': 16,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 136000
    },
    # 2
    {
        'batch_size': 1,
        'x': 41,
        'y': 1600,
        'z': 1408,
        'in_channels': 16,
        'out_channels': 32,
        'kernel_size': (3, 3, 3),
        'strides': (2, 2, 2),
        'paddings': (1, 1, 1),
        'diff': 1e-3,
        'nnz': 136000
    },
    # 3
    {
        'batch_size': 1,
        'x': 21,
        'y': 800,
        'z': 704,
        'in_channels': 3,
        'out_channels': 6,
        'kernel_size': (1, 1, 1),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 20
    },
    # 4
    {
        'batch_size': 1,
        'x': 21,
        'y': 800,
        'z': 704,
        'in_channels': 32,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (2, 2, 2),
        'paddings': (1, 1, 1),
        'diff': 1e-3,
        'nnz': 220939
    },
    # 5
    {
        'batch_size': 1,
        'x': 11,
        'y': 400,
        'z': 352,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 146376
    },
    # 6
    {
        'batch_size': 1,
        'x': 11,
        'y': 400,
        'z': 352,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 146376
    },
    # 7
    {
        'batch_size': 1,
        'x': 5,
        'y': 200,
        'z': 176,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (2, 2, 2),
        'paddings': (0, 1, 1),
        'diff': 1e-3,
        'nnz': 65421
    },
    # 8
    {
        'batch_size': 1,
        'x': 5,
        'y': 200,
        'z': 176,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 65421
    },
    # 9
    {
        'batch_size': 1,
        'x': 5,
        'y': 200,
        'z': 176,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 1, 1),
        'strides': (2, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-2,
        'nnz': 65421
    },
    #10
    {
        'batch_size': 1,
        'x': 5,
        'y': 200,
        'z': 176,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 65421
    },
    #11
    {
        'batch_size': 1,
        'x': 21,
        'y': 800,
        'z': 704,
        'in_channels': 16,
        'out_channels': 16,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 220939
    },
]
class TestSparseConv(unittest.TestCase):

    def test_conv3d(self):
        paddle.seed(0)
        with _test_eager_guard():

            i=0
            values, indices = generate_data(config[i])

            p_shape = [
                config[i]['batch_size'], config[i]['x'], config[i]['y'],
                config[i]['z'], config[i]['in_channels']
            ]
            p_indices = paddle.to_tensor(indices, dtype='int32')
            p_indices = paddle.transpose(p_indices, perm=[1, 0])
            p_values = paddle.to_tensor(values)
            p_input = sparse.sparse_coo_tensor(p_indices, p_values, p_shape, False)
            p_input = sparse.coalesce(p_input)

            p_conv = sparse.nn.Conv3D(
                in_channels=config[i]['in_channels'],
                out_channels=config[i]['out_channels'],
                kernel_size=config[i]['kernel_size'],
                stride=config[i]['strides'],
                padding=config[i]['paddings'],
                bias_attr=False)

            device = torch.device("cuda")
            spnn_conv = spnn.Conv3d(in_channels=config[i]['in_channels'],
                                    out_channels=config[i]['out_channels'],
                                    kernel_size=config[i]['kernel_size'])
            p_conv_weight_torch_reshaped = torch.reshape(torch.tensor(p_conv.weight.numpy()), (-1, config[i]['in_channels'], config[i]['out_channels']))
            spnn_conv.kernel = torch.nn.Parameter(p_conv_weight_torch_reshaped)
            assert np.array_equal(
                p_conv_weight_torch_reshaped.detach().numpy(), spnn_conv.kernel.detach().numpy())
            coords = torch.tensor(indices ,dtype=torch.int32)
            coords = torch.cat((torch.index_select(coords, 1, torch.LongTensor([0,3])), torch.index_select(coords, 1, torch.LongTensor([1, 2]))), dim=1)
            feats = torch.tensor(values, dtype=torch.float32)
            spnn_input = SparseTensor(coords=coords, feats=feats)
            print(spnn_input.coords)
            print(spnn_input.feats)
            spnn_output = spnn_conv(spnn_input)
            print(spnn_output.coords)
            print(spnn_output.feats)
            

if __name__ == "__main__":
    unittest.main()
