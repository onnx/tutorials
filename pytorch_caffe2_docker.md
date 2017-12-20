# Caffe2/PyTorch Docker

Docker images (CPU-only and GPU versions) with ONNX, PyTorch, and Caffe2 are availiable for quickly trying tutorials that use ONNX. To quickly try CPU-only version, simply run:

```
docker run -it --rm onnx/onnx-docker:cpu /bin/bash
```

To run the version with GPU support, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is needed. Execute:
```
nvidia-docker run -it --rm onnx/onnx-docker:gpu /bin/bash
```
