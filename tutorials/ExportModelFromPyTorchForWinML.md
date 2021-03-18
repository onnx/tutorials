<!--- SPDX-License-Identifier: Apache-2.0 -->

# Export PyTorch models for Windows ML

[Windows Machine Learning](https://docs.microsoft.com/windows/ai/windows-ml/) makes it easy to integrate AI into your Windows applications using ONNX models.

## Step 1: Determine the ONNX version your model needs to be in
This depends on which releases of Windows you are targeting. Newer releases of Windows support newer versions of ONNX. This [page](https://docs.microsoft.com/windows/ai/windows-ml/onnx-versions) lists the opset versions supported by different releases of Windows. ONNX 1.2 (opset 7) is the lowest one supported and will work on all versions of Windows ML. Newer versions of ONNX support more types of models.

## Step 2: Export your PyTorch model to that ONNX version
PyTorch's ONNX export support is documented [here](https://pytorch.org/docs/stable/onnx.html). As of PyTorch 1.2, the `torch.onnx.export` function takes a parameter that lets you specify the ONNX opset version.

```python
import torch
import torchvision

dummy_in = torch.randn(10, 3, 224, 224)
model = torchvision.models.resnet18(pretrained=True)

in_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
out_names = [ "output1" ]

torch.onnx.export(model, dummy_in, "resnet18.onnx", input_names=in_names, output_names=out_names, opset_version=7, verbose=True)
```

## Step 3: Integrate the ONNX model into your Windows app
Follow the [tutorials and documentation](https://docs.microsoft.com/windows/ai/windows-ml/) to start using the model in your application. You can code directly against the [Windows ML APIs](https://docs.microsoft.com/windows/ai/windows-ml/integrate-model) or use the [mlgen tool](https://docs.microsoft.com/windows/ai/windows-ml/mlgen) to automatically generate wrapper classes for you.
