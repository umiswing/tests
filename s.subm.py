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
import spconv.pytorch as spconv
import torch
from spconv.core import ConvAlgo
import random
import time

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


torch.set_default_dtype(torch.float16)



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

i = 5
features, indices = generate_data(config[i]) 
device = torch.device("cuda")
features = torch.tensor(features, device=device)
indices = torch.tensor(indices, device=device, dtype=torch.int32)
spatial_shape = [config[i]['x'], config[i]['y'], config[i]['z']]
batch_size = config[i]['batch_size']
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x.features.requires_grad_()
conv = spconv.SubMConv3d(config[i]['in_channels'],config[i]['out_channels'],kernel_size=config[i]['kernel_size'], stride=config[i]['strides'], padding=config[i]['paddings'], dilation=1, bias=False)
conv.to(device=device)

torch.cuda.synchronize(device=device)
for i in range(500):
    y = conv(x)
    y.features.backward(y.features)
torch.cuda.synchronize(device=device)