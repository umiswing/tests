import paddle
import paddle.nn as nn
import paddle.incubate.sparse as sparse
from paddle.fluid.framework import _test_eager_guard
import time
import numpy as np
import torch
import spconv.pytorch as spconv
import inspect

class MiddleExtractor(paddle.nn.Layer):
    def __init__(self,
                 use_norm=True,
                 num_input_features=128,
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        self.middle_conv = paddle.nn.Sequential(
            #nn.Pad3D(1),
            nn.Conv3D(num_input_features, 64, 3, stride=(2, 1, 1), data_format='NDHWC')
        )
    def forward(self, x):
        return self.middle_conv(x)


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class SpconvMiddleExtractor(torch.nn.Module):
    def __init__(self,
                #output_shape,
                use_norm=True,
                num_input_features=128,
                num_filters_down1=[64],
                num_filters_down2=[64, 64],
                name='SpconvMiddleExtractor'):
        super(SpconvMiddleExtractor, self).__init__()

        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False))
            if use_norm:
                #middle_layers.append(BatchNorm1d(o))
                middle_layers.append(torch.nn.BatchNorm1d(o, eps=1e-3, momentum=0.01))
            middle_layers.append(torch.nn.ReLU())

        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))


        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False))
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(spconv.ToDense())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, x):
        out = self.middle_conv(x)
        return out

class SparseMiddleExtractor(paddle.nn.Layer):
    def __init__(self,
                #output_shape,
                use_norm=True,
                num_input_features=128,
                num_filters_down1=[64],
                num_filters_down2=[64, 64],
                name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name

        middle_layers = []
        num_filters = [num_input_features] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d1:
            middle_layers.append(sparse.nn.SubmConv3D(i, o, 3, bias_attr=False))

        middle_layers.append(sparse.nn.Conv3D(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias_attr=False))

        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2

        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d2:
            middle_layers.append(sparse.nn.SubmConv3D(i, o, 3, bias_attr=False))

        middle_layers.append(sparse.nn.Conv3D(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias_attr=False))

        self.middle_conv = nn.Sequential(*middle_layers)

    def forward(self, x):
        sparse_out = self.middle_conv(x)
        #return sparse_out
        return sparse_out.to_dense()


def test():
    paddle.seed(0)
    with _test_eager_guard():
        in_channels = 128 
        # Note: 1. paddle的BatchNorm1D的输入shape不能太大，否则报CUDNN_STATUS_NOT_SUPPORTED.
        shape = [20, 40, 100]
        batch_size = 1
        sparsity = 0.95

        full_shape = [batch_size] + shape + [in_channels]
        print(full_shape)

        total_elements = np.prod(shape)
        nnz = int(total_elements * (1-sparsity))
        print("nnz=", nnz)

        #product indices
        indices = []
        for i in range(4):
           indices.append(paddle.randint(0, full_shape[i], [1, nnz])) 

        indices = paddle.concat(indices)
        #product values
        values = paddle.randn((nnz, in_channels))

        sparse_x = sparse.sparse_coo_tensor(indices, values, shape=full_shape)
        sparse_x = paddle.incubate.sparse.coalesce(sparse_x)

        dense_x = sparse_x.to_dense()

        #spconv
        device = torch.device("cuda")
        torch_x = torch.tensor(dense_x.numpy(), device=device)

        spconv_x = spconv.SparseConvTensor.from_dense(torch_x)

        #whether to use batch_norm
        use_norm = False

        dense_model = MiddleExtractor(use_norm=use_norm, num_input_features=in_channels)
        spconv_model = SpconvMiddleExtractor(use_norm=use_norm, num_input_features=in_channels).to(device)
        sparse_model = SparseMiddleExtractor(use_norm=use_norm, num_input_features=in_channels)
        layer_nums = len(sparse_model.middle_conv)
        block_size = 3 if use_norm else 2
        layer_nums = int(layer_nums / block_size)

        for i in range(0, layer_nums):
            weight = paddle.to_tensor(spconv_model.middle_conv[i * block_size].weight.detach().cpu().numpy())
            sparse_model.middle_conv[i * block_size].weight.set_value(paddle.transpose(paddle.to_tensor(weight), [1,2,3,4,0]))
            if use_norm:
               bn_weight = paddle.to_tensor(spconv_model.middle_conv[i*block_size + 1].weight.detach().cpu().numpy()) 
               sparse_model.middle_conv[i * block_size + 1].weight.set_value(bn_weight)

        print(dense_model)
        print(sparse_model)
        print(spconv_model)
        paddle.device.cuda.synchronize()

        #padde dense

        #padde sparse
        sparse_x.stop_gradient=True
        out2 = sparse_model(sparse_x)
        paddle.device.cuda.synchronize()
        sparse_x.stop_gradient=False
        t2 = time.perf_counter()
        for i in range(100):
            out2 = sparse_model(sparse_x)
            #out2.backward(out2)
        paddle.device.cuda.synchronize()
        t3 = time.perf_counter()

        #spconv
        spconv_x.features.required_grad = False
        out3 = spconv_model(spconv_x)
        #out3.backward(out3)
        spconv_x.features.required_grad=True
        spconv_x.features.requires_grad_()
        torch.cuda.synchronize(device)
        t4 = time.perf_counter()
        for i in range(100):
            out3 = spconv_model(spconv_x)
            #out3.backward(out3)
        t5 = time.perf_counter()

        # Note 2. sparse的BatchNorm底层是使用paddle.nn.BatchNorm1D对values进行bn计算,测试发现BatchNorm1D的性能比BatchNorm3D差，因此use_norm=True的情况，需要更高的稀疏度才能比dense的快
        # Note 3. 只跑前向，sparse的耗时和spconv接近，稀疏度越高sparse的性能越好，当前方式测试前向+反向，spconv的耗时很高, 原因未知
        print("sparse time: ", t3 - t2)
        print("spconv time: ", t5 - t4)

        # Note 4. paddle和torch的BN存在误差，测试shape=(4000, 64)的随机输入，单层BN前向误差在1e-6, 反向误差在1e-4
        #verify the forward calculation result
        assert np.allclose(paddle.transpose(out2, [0, 4, 1, 2, 3]).numpy(), out3.detach().cpu().numpy(), atol=1e-2, rtol=1e-2)
        #print(out2)

        #verify the backward calculation result
        #assert np.allclose(spconv_x.features.grad.cpu().numpy(),
        #sparse_x.grad.values().numpy(), atol=1e-3, rtol=1e-3)

test()
