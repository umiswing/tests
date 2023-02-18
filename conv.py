# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import paddle.incubate.sparse as sparse
import paddle.incubate as pi
import time

paddle.set_default_dtype("float16")
torch.set_default_dtype(torch.float16)


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
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-2,
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
        'in_channels': 32,
        'out_channels': 32,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-2,
        'nnz': 220939
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
        'diff': 1e-2,
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
        'diff': 1e-2,
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
        'diff': 1e-2,
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
        'diff': 1e-3,
        'nnz': 65421
    },
    #10
    {
        'batch_size': 1,
        'x': 21,
        'y': 800,
        'z': 704,
        'in_channels': 32,
        'out_channels': 32,
        'kernel_size': (1, 1, 1),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
        'nnz': 4002
    },
]
class SparseMiddleExtractor(paddle.nn.Layer):
    def __init__(self,
                i,
                name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name

        middle_layers = []
        middle_layers.append(sparse.nn.Conv3D(in_channels=config[i]['in_channels'], out_channels=config[i]['out_channels'], kernel_size=config[i]['kernel_size'],
                                              stride=config[i]['strides'], padding=config[i]['paddings'], bias_attr=False))

        self.middle_conv = nn.Sequential(*middle_layers)

    def forward(self, x):
        sparse_out = self.middle_conv[0](x,cutlass=True)
        return sparse_out
        #return sparse_out.to_dense()

class SpconvMiddleExtractor(torch.nn.Module):
    def __init__(self,
                i,
                name='SpconvMiddleExtractor'):
        super(SpconvMiddleExtractor, self).__init__()

        middle_layers = []

        middle_layers.append(spconv.SparseConv3d(config[i]['in_channels'],
                                                 config[i]['out_channels'],
                                                 kernel_size=config[i]['kernel_size'],
                                                 stride=config[i]['strides'],
                                                 padding=config[i]['paddings'],
                                                 dilation=1,
                                                 bias=False))
        #middle_layers.append(spconv.ToDense())

        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, x):
        out = self.middle_conv(x)
        return out

class TestSparseConv(unittest.TestCase):

    def test_conv3d(self):
        paddle.seed(0)
        with _test_eager_guard():

            i = 8
            values, indices = generate_data(config[i])

            p_shape = [
                config[i]['batch_size'], config[i]['x'], config[i]['y'],
                config[i]['z'], config[i]['in_channels']
            ]
            p_indices = paddle.to_tensor(indices, dtype='int32')
            p_indices = paddle.transpose(p_indices, perm=[1, 0])
            p_values = paddle.to_tensor(values)
            p_input = pi.sparse.sparse_coo_tensor(p_indices, p_values, p_shape)

            p_input = paddle.incubate.sparse.coalesce(p_input)
            p_conv = pi.sparse.nn.SubmConv3D(
                in_channels=config[i]['in_channels'],
                out_channels=config[i]['out_channels'],
                kernel_size=config[i]['kernel_size'],
                stride=config[i]['strides'],
                padding=config[i]['paddings'],
                bias_attr=False)
            #p_conv = pi.sparse.nn.Conv3D(in_channels=config[i]['in_channels'], out_channels=config[i]['out_channels'], kernel_size=config[i]['kernel_size'],
                                         #stride=config[i]['strides'], padding=config[i]['paddings'], bias_attr=False)

            device = torch.device("cuda")
            spatial_shape = [config[i]['x'], config[i]['y'], config[i]['z']]
            s_values = torch.tensor(np.array(p_input.values()), device=device)
            s_indices = torch.tensor(np.array(
                paddle.transpose(p_input.indices(), perm=[1, 0])),
                                     device=device).int()
            s_input = spconv.SparseConvTensor(s_values, s_indices,
                                              spatial_shape,
                                              config[i]['batch_size'])
            #s_conv = spconv.SparseConv3d(config[i]['in_channels'],
            #config[i]['out_channels'],
            #kernel_size=config[i]['kernel_size'],
            #stride=config[i]['strides'],
            #padding=config[i]['paddings'],
            #dilation=1,
            #bias=False)
            s_conv = spconv.SubMConv3d(config[i]['in_channels'],
                                       config[i]['out_channels'],
                                       kernel_size=config[i]['kernel_size'],
                                       stride=config[i]['strides'],
                                       padding=config[i]['paddings'],
                                       dilation=1,
                                       bias=False)
                                       #algo=ConvAlgo.Native)
            s_conv.to(device=device)

            s_conv.weight = torch.nn.Parameter(torch.tensor(
            np.transpose(p_conv.weight.numpy(), (4, 0, 1, 2, 3))).cuda().contiguous())

            #spconv_model = SpconvMiddleExtractor(i).to(device)
            #sparse_model = SparseMiddleExtractor(i)

            #weight = paddle.to_tensor(spconv_model.middle_conv[0].weight.detach().cpu().numpy())
            #sparse_model.middle_conv[0].weight.set_value(paddle.transpose(paddle.to_tensor(weight), [1,2,3,4,0]))

            #print(sparse_model)
            #print(spconv_model)
            #p_input.stop_gradient = False
            #paddle.device.cuda.synchronize()
            #p_out = sparse_model(p_input)
            #s_out = spconv_model(s_input)
            #p_out.backward(p_out)
            #s_out.backward(s_out)
            #assert np.allclose(paddle.transpose(p_out, [0, 4, 1, 2, 3]).numpy(), s_out.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)
            #s_out_features_nd = s_out.features.cpu().detach().numpy().flatten()
            #p_out_features_nd = p_out.values().numpy().flatten()
            #assert np.allclose(s_out_features_nd,
            #p_out_features_nd, atol=config[i]['diff'], rtol=config[i]['diff'])


            c_out = p_conv(p_input, True)
            paddle.device.cuda.synchronize()
            p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()
            print(p_out)
            print(c_out)
            #p_out = paddle.incubate.sparse.coalesce(p_out)
            #c_out = paddle.incubate.sparse.coalesce(c_out)
            s_out = s_conv(s_input)
            #s_out.backward(s_out)
            torch.cuda.synchronize(device=device)
            #print(s_out.features)
            assert np.array_equal(c_out.indices().numpy(),
                                  p_out.indices().numpy())
            assert np.allclose(c_out.values().numpy().flatten(),
                               p_out.values().numpy().flatten(),
                               atol=config[i]['diff'],
                               rtol=config[i]['diff'])
            assert np.array_equal(
            s_out.indices.cpu().detach().numpy().transpose(1, 0), p_out.indices().numpy())

            s_out_features_nd = s_out.features.cpu().detach().numpy().flatten()
            p_out_features_nd = p_out.values().numpy().flatten()
            #print(s_out.indices)
            #print(s_out.features)
            #print('\n')
            #print(p_out)
            #print(s_out_features_nd)
            #print(p_out_features_nd)

            assert np.allclose(s_out_features_nd,
            p_out_features_nd, atol=config[i]['diff'], rtol=config[i]['diff'])

            paddle.device.cuda.synchronize()
            for n in range(100):
                c_out = p_conv(p_input, True)
            paddle.device.cuda.synchronize()

            t0 = time.perf_counter()
            for n in range(500):
                c_out = p_conv(p_input, True)
            paddle.device.cuda.synchronize()
            t1 = time.perf_counter()

            paddle.device.cuda.synchronize()
            for n in range(100):
                p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()

            t2 = time.perf_counter()
            for n in range(500):
                p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()
            t3 = time.perf_counter()

            torch.cuda.synchronize(device=device)
            for n in range(100):
                s_out = s_conv(s_input)
            torch.cuda.synchronize(device=device)

            t4 = time.perf_counter()
            for n in range(500):
                s_out = s_conv(s_input)
            torch.cuda.synchronize(device=device)
            t5 = time.perf_counter()

            print("cutlass time:", t1 - t0)
            print("paddle time:", t3 - t2)
            print("spconv time:", t5 - t4)

if __name__ == "__main__":
    unittest.main()
