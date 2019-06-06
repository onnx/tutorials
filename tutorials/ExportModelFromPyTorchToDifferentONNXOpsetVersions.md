### Export model from PyTorch to ONNX:

Default version for ONNX model is 9 when it is exported from PyTorch.

```python
import torch
import torchvision

# No CUDA
dummy_input = torch.randn(10, 3, 224, 224)
model = torchvision.models.resnet18(pretrained=True)

# CUDA
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)
```

### Version Conversion

Version Conversion for BatchNormalization from opset 8 to 9:

<img src="assets/batchnorm.png" />


### Downgrade Version Conversion from 9 to 8:

```python
import onnx

# Load the model
model = onnx.load("path_to/resnet18.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

from onnx import version_converter

# Convert to version 8
converted_model = version_converter.convert_version(onnx_model, 8)

# Save model
onnx.save(converted_model, "path_to/resnet18_v8.onnx")
```

### Upgrade Version Conversion from 8 to 9

```python
# Convert to version 9
converted_model9 = version_converter.convert_version(converted_model, 9)

# Save model
onnx.save(converted_model9, "path_to/resnet18_v9.onnx")
```

### Downgrade Version Conversion from 8 to 7

```python
# Convert to version 7
converted_model7 = version_converter.convert_version(converted_model, 7)

# Save model
onnx.save(converted_model7, "path_to/resnet18_v7.onnx")
```

### Upgrade Version Conversion from 7 to 9

```python
# Convert to version 9
converted_model79 = version_converter.convert_version(converted_model7, 9)

# Save model
onnx.save(converted_model79, "path_to/resnet18_v79.onnx")
```