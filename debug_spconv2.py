import torch
import paddle
import numpy as np
import spconv
import spconv.pytorch as spconv

def load_data(indices_path, values_path):
    dense_shape = [2, 2304, 2304, 32]
    paddle_indices = paddle.to_tensor(np.load(indices_path), stop_gradient=False).cuda()
    paddle_values = paddle.to_tensor(np.load(values_path),  stop_gradient=False).cuda()
    torch_indice = torch.from_numpy(np.load(indices_path)).cuda()
    torch_values = torch.from_numpy(np.load(values_path)).cuda()
    # print("\nIndices: ", indices)
    # print("\nValues: ", values)

    paddle_sp_tensor = paddle.sparse.sparse_coo_tensor(
        indices=paddle_indices.transpose((1, 0)),
        values=paddle_values,
        shape=dense_shape,
        place=paddle_values.place,
        dtype=paddle.float32,
    )
    # paddle_sp_tensor.indices() == paddle_indices.T
    # paddle_sp_tensor.values() == paddle_values

    torch_sp_tensor = spconv.SparseConvTensor(
        features=torch_values,
        indices=torch_indice,
        spatial_shape=(dense_shape[1], dense_shape[2]),
        batch_size=dense_shape[0]
    )
    # torch_sp_tensor.features == torch_values
    # torch_sp_tensor.indices == torch_indice

    return paddle_sp_tensor, torch_sp_tensor 


class PaddleBaseConv2D3x3(paddle.nn.Layer):
    def __init__(
        self, in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias_attr=True
    ):
        super(PaddleBaseConv2D3x3, self).__init__()

        self.spconv2d = paddle.sparse.nn.SubmConv2D(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=bias_attr,
            key=indice_key,
        )
        self.bn = paddle.sparse.nn.BatchNorm(out_planes, epsilon=1e-3, momentum=1 - 0.01, data_format='NHWC')

    def forward(self, x):
        out = self.spconv2d(x)
        out_bn = self.bn(out)
        return out, out_bn

class TorchBaseConv2D3x3(torch.nn.Module):
    def __init__(
        self, in_planes, out_planes, stride=1, dilation=1, indice_key=None, bias=True
    ):
        super(TorchBaseConv2D3x3, self).__init__()
        self.spconv2d = spconv.SubMConv2d(
                            in_planes,
                            out_planes,
                            kernel_size=3,
                            stride=stride,
                            dilation=dilation,
                            padding=1,
                            bias=bias,
                            indice_key=indice_key
        )
        self.bn = spconv.SparseSequential(
            torch.nn.BatchNorm1d(out_planes, eps=1e-03, momentum=0.01)
        )

    def forward(self, x):
        out = self.spconv2d(x)
        out_bn = self.bn(out)
        return out, out_bn


def print_weight_diff(demo_paddle_layer, demo_torch_layer):
    paddle_weight_before_forward = np.transpose(demo_paddle_layer.spconv2d.weight.numpy(), (3, 0, 1, 2))
    torch_weight_before_forward = demo_torch_layer.spconv2d.weight.data.cpu().detach().numpy()
    print('spconv2d.weight are close:', np.allclose(paddle_weight_before_forward, torch_weight_before_forward))
    print('max_diff: ', abs(paddle_weight_before_forward - torch_weight_before_forward).max())

    paddle_bias_before_forward = demo_paddle_layer.spconv2d.bias.numpy()
    torch_bias_before_forward = demo_torch_layer.spconv2d.bias.data.cpu().detach().numpy()
    print('spconv2d.bias are close:', np.allclose(paddle_bias_before_forward, torch_bias_before_forward))
    print('max_diff: ', abs(paddle_bias_before_forward - torch_bias_before_forward).max())


def print_output_diff(demo_torch_layer_out, demo_torch_layer_bn_out, demo_paddle_layer_out, demo_paddle_layer_bn_out):
    # to dense
    demo_torch_layer_out = demo_torch_layer_out.dense().cpu().detach().numpy()
    demo_paddle_layer_out =  paddle.transpose(demo_paddle_layer_out.to_dense(), perm=[0, 3, 1, 2]).numpy()

    demo_torch_layer_bn_out = demo_torch_layer_bn_out.dense().cpu().detach().numpy()
    demo_paddle_layer_bn_out =  paddle.transpose(demo_paddle_layer_bn_out.to_dense(), perm=[0, 3, 1, 2]).numpy()

    print('spconv output are close:', np.allclose(demo_torch_layer_out, demo_paddle_layer_out, rtol=0, atol=1e-7))
    print('max_diff: ', abs(demo_torch_layer_out - demo_paddle_layer_out).max())
    non_zero_indices = (demo_torch_layer_out - demo_paddle_layer_out) != 0
    non_zero_values = (demo_torch_layer_out - demo_paddle_layer_out)[non_zero_indices]
    large_diff_indices = (demo_torch_layer_out - demo_paddle_layer_out) > 1
    large_diff_values = (demo_torch_layer_out - demo_paddle_layer_out)[large_diff_indices]
    print('tensor_shape: ', demo_torch_layer_out.shape, 
          ', all_diff_value_num: ', non_zero_values.shape, 
          ', large_diff_value_num:', large_diff_values.shape)

    print('bn output are close:', np.allclose(demo_torch_layer_bn_out, demo_paddle_layer_bn_out, rtol=0, atol=1e-7))
    print('max_diff: ', abs(demo_torch_layer_bn_out - demo_paddle_layer_bn_out).max())


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    paddle.seed(seed)
    torch.manual_seed(seed)

    torch.set_printoptions(precision=8, sci_mode=False)
    np.set_printoptions(precision=8, suppress=True)

    indices_path = ''
    values_path = ''
    paddle_sp_tensor, torch_sp_tensor = load_data(indices_path, values_path)

    # layer
    demo_paddle_layer = PaddleBaseConv2D3x3(32, 32, indice_key=None, bias_attr=True)
    demo_torch_layer = TorchBaseConv2D3x3(32, 32, indice_key=None, bias=True)

    # torch spconv2d load paddle spconv2d weight, bias
    demo_torch_layer.spconv2d.weight.data= torch.from_numpy(np.transpose(demo_paddle_layer.state_dict()['spconv2d.weight'].numpy(), (3, 0, 1, 2))).contiguous()
    demo_torch_layer.spconv2d.bias.data = torch.from_numpy(demo_paddle_layer.state_dict()['spconv2d.bias'].numpy())

    # torch bn load paddle bn weight, bias
    demo_torch_layer.state_dict()['bn.0.weight'].data = torch.from_numpy(demo_paddle_layer.state_dict()['bn.weight'].numpy())
    demo_torch_layer.state_dict()['bn.0.bias'].data = torch.from_numpy(demo_paddle_layer.state_dict()['bn.bias'].numpy())
    demo_torch_layer.state_dict()['bn.0.running_mean'].data = torch.from_numpy(demo_paddle_layer.state_dict()['bn._mean'].numpy())
    demo_torch_layer.state_dict()['bn.0.running_var'].data = torch.from_numpy(demo_paddle_layer.state_dict()['bn._variance'].numpy())

    print('---- before forward:')
    print_weight_diff(demo_paddle_layer, demo_torch_layer)

    print('\n---- forward:')
    # forward

    demo_torch_layer = demo_torch_layer.cuda()

    demo_torch_layer_out, demo_torch_layer_bn_out = demo_torch_layer(torch_sp_tensor)
    demo_paddle_layer_out, demo_paddle_layer_bn_out = demo_paddle_layer(paddle_sp_tensor)    

    print_output_diff(demo_torch_layer_out, demo_torch_layer_bn_out, demo_paddle_layer_out, demo_paddle_layer_bn_out)

    print('\n---- after forward:')
    # without backward, parameters has not been updated.
    print_weight_diff(demo_paddle_layer, demo_torch_layer)
