import torch


def register_custom_op():
    def my_group_norm(g, input, num_groups, scale, bias, eps):
        return g.op("mydomain::testgroupnorm", input, num_groups, scale, bias, epsilon_f=0.)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('mynamespace::custom_group_norm', my_group_norm, 5)


def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x, num_groups, scale, bias):
            return torch.ops.mynamespace.custom_group_norm(x, num_groups, scale, bias, 0.)

    X = torch.randn(3, 2, 1, 2)
    num_groups = torch.tensor([2.])
    scale = torch.tensor([1., 1.])
    bias = torch.tensor([0., 0.])
    inputs = (X, num_groups, scale, bias)

    f = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                       opset_version=9,
                       example_outputs=None,
                       input_names=["X", "num_groups", "scale", "bias"], output_names=["Y"])



torch.ops.load_library(
    "build/lib.linux-x86_64-3.7/custom_group_norm.cpython-37m-x86_64-linux-gnu.so")
register_custom_op()
export_custom_op()
