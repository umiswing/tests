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
#from paddle.fluid.framework import _test_eager_guard
import random
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
import torch
import logging
import paddle.sparse as sparse
import paddle.incubate as pi
import time
import sys
import numpy as np
from numpy.linalg import norm
import math

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
        'diff': 1e-2,
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
        'diff': 1e-2,
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
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': (3, 3),
        'strides': (1, 1),
        'paddings': (0, 0),
        'diff': 1e-2,
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
        'nnz': 55521
    },
    #11
    {
        'batch_size': 1,
        'x': 11,
        'y': 400,
        'z': 352,
        'in_channels': 128,
        'out_channels': 128,
        'kernel_size': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'diff': 1e-2,
        'nnz': 146376
    },    
]
class TestSparseConv(unittest.TestCase):

    def test_conv3d(self):
        paddle.seed(0)
#        with _test_eager_guard():

        i=6
        values, indices = generate_data(config[i])

        p_shape = [
            config[i]['batch_size'], config[i]['x'], config[i]['y'],
            config[i]['in_channels']
        ]
        p_indices = paddle.to_tensor(indices, dtype='int32')
        p_indices = paddle.transpose(p_indices, perm=[1, 0])
        p_values = paddle.to_tensor(values)
        p_input = sparse.sparse_coo_tensor(p_indices, p_values, p_shape, False)
        p_input = sparse.coalesce(p_input)
        p_input = core.eager.sparse_coo_tensor(p_input.indices(), p_input.values(), p_shape, False)

        p_conv = sparse.nn.Conv2D(
            in_channels=config[i]['in_channels'],
            out_channels=config[i]['out_channels'],
            kernel_size=config[i]['kernel_size'],
            stride=config[i]['strides'],
            padding=config[i]['paddings'],
            bias_attr=False)

        device = torch.device("cuda")
        spatial_shape = [config[i]['x'], config[i]['y']]
        s_values = torch.tensor(np.array(p_input.values()), device=device)
        s_indices = torch.tensor(np.array(
            paddle.transpose(p_input.indices(), perm=[1, 0])),
                                 device=device).int()
        s_input = spconv.SparseConvTensor(s_values, s_indices,
                                          spatial_shape,
                                          config[i]['batch_size'])
        s_conv = spconv.SparseConv2d(config[i]['in_channels'],
                                   config[i]['out_channels'],
                                   kernel_size=config[i]['kernel_size'],
                                   stride=config[i]['strides'],
                                   padding=config[i]['paddings'],
                                   dilation=1,
                                   bias=False)
        s_conv.to(device=device)

        s_conv.weight = torch.nn.Parameter(torch.tensor(
        np.transpose(p_conv.weight.numpy(), (3, 0, 1, 2))).cuda().contiguous())

        c_out = p_conv(p_input)
        c_out.backward(c_out)
        paddle.device.cuda.synchronize()
        s_input.features.requires_grad_()
        s_out = s_conv(s_input)
        s_out.features.backward(s_out.features)
        torch.cuda.synchronize(device=device)

        # convert spconv.SparseConvTensor to sparse.sparse_coo_tensor
        s_to_pd_indices = paddle.to_tensor(s_out.indices.cpu().detach().numpy().transpose(1, 0).tolist(), dtype='int32')
        s_to_pd_values = paddle.to_tensor(s_out.features.cpu().detach().numpy().tolist())
        s_to_pd = sparse.sparse_coo_tensor(s_to_pd_indices, s_to_pd_values, [s_out.batch_size] + s_out.spatial_shape + [config[i]['out_channels']], False)
        s_to_pd = sparse.coalesce(s_to_pd)
        # out indices
        #assert np.array_equal(
        #    s_to_pd.indices().numpy(), c_out.indices().numpy())

        #assert np.allclose(s_to_pd.values().numpy().flatten(),
        #                   c_out.values().numpy().flatten(), atol=config[i]['diff'], rtol=config[i]['diff'])

        s_input_grad_np = s_input.features.grad.cpu().numpy()
        p_input_grad_np = p_input.grad.values().numpy()
        print(s_input_grad_np.shape)
        print(p_input_grad_np.shape)
        s_weight_grad_np = s_conv.weight.grad.cpu().detach().numpy()
        p_weight_grad_np = np.transpose(p_conv.weight.grad.numpy(), (3, 0, 1, 2))
        print(s_weight_grad_np.shape)
        print(p_weight_grad_np.shape)

        for i in range(s_input_grad_np.shape[0]):
            s = s_input_grad_np[i]
            p = p_input_grad_np[i]
            cos = np.dot(s, p)/(norm(s)*norm(p))
            if cos < 0.999 or math.isnan(cos):
              print("========")
              print(cos)
              print(s)
              print(p)
              print(np.dot(s,p))
              print(norm(s))
              print(norm(p))
        
        for i in range(s_weight_grad_np.shape[0]):
            for j in range(s_weight_grad_np.shape[1]):
                for k in range(s_weight_grad_np.shape[2]):
                    s = s_weight_grad_np[i,j,k]
                    p = p_weight_grad_np[i,j,k]
                    cos = np.dot(s, p)/(norm(s)*norm(p))
                    if cos < 0.999 or math.isnan(cos):
                      print("========")
                      print(cos)
                      print(s)
                      print(p)
                      print(np.dot(s,p))
                      print(norm(s))
                      print(norm(p))

if __name__ == "__main__":
    unittest.main()
