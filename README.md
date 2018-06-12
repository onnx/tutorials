# [ONNX](https://github.com/onnx/onnx) tutorials

## Importing and Exporting from frameworks

| Framework / tool | Installation | Exporting to ONNX (frontend) | Importing ONNX models (backend) |
| --- | --- | --- | --- |
| [Caffe2](http://caffe2.ai) | [part of caffe2 package](https://github.com/caffe2/caffe2/tree/master/caffe2/python/onnx) | [Exporting](tutorials/Caffe2OnnxExport.ipynb) | [Importing](tutorials/OnnxCaffe2Import.ipynb) |
| [PyTorch](http://pytorch.org/) | [part of pytorch package](http://pytorch.org/docs/master/onnx.html) | [Exporting](tutorials/PytorchOnnxExport.ipynb), [Extending support](tutorials/PytorchAddExportSupport.md) | coming soon |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Exporting](tutorials/CntkOnnxExport.ipynb) | [Importing](tutorials/OnnxCntkImport.ipynb) |
| [Apache MXNet](http://mxnet.incubator.apache.org/) | part of mxnet package [docs](http://mxnet.incubator.apache.org/versions/1.2.0/api/python/contrib/onnx.html) [github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | coming soon | [Importing](tutorials/OnnxMxnetImport.ipynb) [experimental] |
| [Chainer](https://chainer.org/) | [chainer/onnx-chainer](https://github.com/chainer/onnx-chainer) | [Exporting](tutorials/ChainerOnnxExport.ipynb) | coming soon |
| [TensorFlow](https://www.tensorflow.org/) | [onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) | [Exporting](tutorials/OnnxTensorflowExport.ipynb) | [Importing](tutorials/OnnxTensorflowImport.ipynb) [experimental] |
| [Apple CoreML](https://developer.apple.com/documentation/coreml) | [onnx/onnx-coreml](https://github.com/onnx/onnx-coreml) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnxmltools) | [Importing](tutorials/OnnxCoremlImport.ipynb) |
| [SciKit-Learn](http://scikit-learn.org/) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnxmltools) | n/a |

## End-to-end tutorials

* [Docker image for Caffe2/PyTorch tutorials](pytorch_caffe2_docker.md)
* [Converting SuperResolution model from PyTorch to Caffe2 and deploying on mobile device](tutorials/PytorchCaffe2SuperResolution.ipynb)
* [Transferring SqueezeNet from PyTorch to Caffe2 and to Android app](tutorials/PytorchCaffe2MobileSqueezeNet.ipynb)
* [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
* [Serving ONNX models with MXNet Model Server](tutorials/ONNXMXNetServer.ipynb)
* [Converting Style Transfer model from PyTorch to CoreML and deploying to an iPhone](https://github.com/onnx/tutorials/tree/master/examples/CoreML/ONNXLive)

## ONNX tools

* [Verifying correctness and comparing performance](tutorials/CorrectnessVerificationAndPerformanceComparison.ipynb)
* [Visualizing an ONNX model](tutorials/VisualizingAModel.md) (useful for debugging)
* [Example of operating on ONNX protobuf](https://github.com/onnx/onnx/blob/master/onnx/examples/Protobufs.ipynb)

## Contributing

We welcome improvements to the convertor tools and contributions of new ONNX bindings. Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

Use ONNX for something cool? Send the tutorial to this repo by submitting a PR.
