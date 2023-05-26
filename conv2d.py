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
import logging
import paddle.sparse as sparse
import paddle.incubate as pi
import time
import sys
from numpy.linalg import norm
import math
from paddle.nn.initializer import Normal

paddle.set_default_dtype("float32")

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
    {
        'batch_size': 1,
        'x': 5,
        'y': 200,
        'in_channels': 32,
        'out_channels': 64,
        'kernel_size': (3, 3),
        'stride': (2,1),
        'padding': (1,0),
        'dilation': (2,1),
        'diff': 1e-2,
        'nnz': 654
    },
]
class TestSparseConv(unittest.TestCase):

    def test_conv3d(self):
        paddle.seed(0)
#        with _test_eager_guard():

        i=0
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

        #bias_attr=paddle.ParamAttr(name='bias',initializer=Normal, learning_rate=0.5,regularizer=paddle.regularizer.L2Decay(1.0),trainable=True)
        p_conv = sparse.nn.Conv2D(
            in_channels=config[i]['in_channels'],
            out_channels=config[i]['out_channels'],
            kernel_size=config[i]['kernel_size'],
            stride=config[i]['stride'],
            padding=config[i]['padding'],
            dilation=config[i]['dilation'],
            bias_attr=False)

        dense_input = p_input.to_dense()
        dense_weight = p_conv.weight
        dense_bias = p_conv.bias
        h = dense_weight.shape[0]
        w = dense_weight.shape[1]
        c = dense_weight.shape[2]
        m = dense_weight.shape[3]
        # h,w,c,m -> m,c,h,w
        dense_weight = paddle.transpose(dense_weight, [3,2,0,1])
        print(p_conv.bias)
        c_out = p_conv(p_input)
        dense_out = nn.functional.conv2d(x=dense_input, weight=dense_weight,data_format="NHWC", bias=dense_bias, stride=config[i]['stride'], padding=config[i]['padding'],dilation=config[i]['dilation'])
        dense_out.backward(c_out.to_dense())
        c_out.backward(c_out)
        paddle.device.cuda.synchronize()
        c_out_numpy = c_out.to_dense().numpy()
        dense_out_numpy = dense_out.numpy()
        print(c_out.shape)
        for hh in range(c_out.shape[1]):
            for ww in range(c_out.shape[2]):
                c = c_out_numpy[0,hh,ww]
                d = dense_out_numpy[0,hh,ww]
                cos = np.dot(c, d)/(norm(c)*norm(d))
                if cos < 0.99999 or math.isnan(cos):
                  print("========")
                  print(cos)
                  print(c)
                  print(d)
                  print(np.dot(c,d))
                  print(norm(c))
                  print(norm(d))
        print(p_conv.weight.grad)
        assert np.allclose(c_out.to_dense().numpy().flatten(), dense_out.numpy().flatten(), rtol=0, atol=1e-2)

if __name__ == "__main__":
    unittest.main()
