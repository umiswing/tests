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
import paddle.sparse as sparse
import paddle.incubate as pi
import time
import sys

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
        'kernel_size': (3, 3, 3),
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
        'in_channels': 32,
        'out_channels': 32,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-3,
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
        'diff': 1e-3,
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
        'nnz': 55521
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

            i=8
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

            p_input = core.eager.sparse_coo_tensor(p_input.indices(), p_input.values(), p_shape, False)
            p_conv = sparse.nn.SubmConv3D(
                in_channels=config[i]['in_channels'],
                out_channels=config[i]['out_channels'],
                kernel_size=config[i]['kernel_size'],
                stride=config[i]['strides'],
                padding=config[i]['paddings'],
                bias_attr=False)

            device = torch.device("cuda")
            spatial_shape = [config[i]['x'], config[i]['y'], config[i]['z']]
            s_values = torch.tensor(np.array(p_input.values()), device=device)
            s_indices = torch.tensor(np.array(
                paddle.transpose(p_input.indices(), perm=[1, 0])),
                                     device=device).int()
            s_input = spconv.SparseConvTensor(s_values, s_indices,
                                              spatial_shape,
                                              config[i]['batch_size'])
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

            c_out = p_conv(p_input)
            c_out.backward(c_out)
            paddle.device.cuda.synchronize()
            s_input.features.requires_grad_()
            s_out = s_conv(s_input)
            s_out.features.backward(s_out.features)
            torch.cuda.synchronize(device=device)
            # out indices
            assert np.array_equal(
            s_out.indices.cpu().detach().numpy().transpose(1, 0), c_out.indices().numpy())

            s_out_features_nd = s_out.features.cpu().detach().numpy().flatten()
            c_out_features_nd = c_out.values().numpy().flatten()

            # out values
            assert np.allclose(s_out_features_nd,
            c_out_features_nd, atol=config[i]['diff'], rtol=config[i]['diff'])
            print(s_input.features.grad)
            print(p_input.grad)
            # x_grad
            assert np.allclose(s_input.features.grad.cpu().detach(
            ).numpy().flatten(), p_input.grad.values().numpy().flatten(), atol=config[i]['diff'], rtol=config[i]['diff'])
            # kernel_grad
            assert np.allclose(s_conv.weight.cpu().detach().numpy().flatten(), np.transpose(
                p_conv.weight.numpy(), (4, 0, 1, 2, 3)).flatten(), atol=config[i]['diff'], rtol=config[i]['diff'])

            paddle.device.cuda.synchronize()
            for n in range(100):
                c_out = p_conv(p_input)
                c_out.backward(c_out)
            paddle.device.cuda.synchronize()

            t0 = time.perf_counter()
            for n in range(500):
                c_out = p_conv(p_input)
                c_out.backward(c_out)
            paddle.device.cuda.synchronize()
            t1 = time.perf_counter()

            torch.cuda.synchronize(device=device)
            for n in range(100):
                s_out = s_conv(s_input)
                s_out.features.backward(s_out.features)
            torch.cuda.synchronize(device=device)

            t4 = time.perf_counter()
            for n in range(500):
                s_out = s_conv(s_input)
                s_out.features.backward(s_out.features)
            torch.cuda.synchronize(device=device)
            t5 = time.perf_counter()

            print("cutlass time:", t1 - t0)
            print("spconv time:", t5 - t4)

if __name__ == "__main__":
    unittest.main()
