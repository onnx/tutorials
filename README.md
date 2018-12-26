# [ONNX](https://github.com/onnx/onnx) tutorials

[Open Neural Network Exchange (ONNX)](http://onnx.ai/) is an open standard format of machine learning models to offer interoperability between various AI frameworks. With ONNX, AI develpers could choose the best framework for training and switch to different one for shipping.

ONNX is supported by [a community of partners](https://onnx.ai/supported-tools), and more and more AI frameworks are buiding ONNX support including PyTorch, Caffe2, Microsoft Cognitive Toolkit and Apache MXNet.  

## Getting ONNX models

* Choose a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models). A lot of pre-trained ONNX models are provided for common scenarios. 
* Convert models from mainstream frameworks. More tutorials are below.

| Framework / tool | Installation | Exporting to ONNX (frontend) | Importing ONNX models (backend) |
| --- | --- | --- | --- |
| [Caffe2](http://caffe2.ai) | [part of caffe2 package](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) | [Exporting](tutorials/Caffe2OnnxExport.ipynb) | [Importing](tutorials/OnnxCaffe2Import.ipynb) |
| [PyTorch](http://pytorch.org/) | [part of pytorch package](http://pytorch.org/docs/master/onnx.html) | [Exporting](tutorials/PytorchOnnxExport.ipynb), [Extending support](tutorials/PytorchAddExportSupport.md) | coming soon |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Exporting](tutorials/CntkOnnxExport.ipynb) | [Importing](tutorials/OnnxCntkImport.ipynb) |
| [Apache MXNet](http://mxnet.incubator.apache.org/) | part of mxnet package [docs](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html) [github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | [Exporting](tutorials/MXNetONNXExport.ipynb) | [Importing](tutorials/OnnxMxnetImport.ipynb) |
| [Chainer](https://chainer.org/) | [chainer/onnx-chainer](https://github.com/chainer/onnx-chainer) | [Exporting](tutorials/ChainerOnnxExport.ipynb) | coming soon |
| [TensorFlow](https://www.tensorflow.org/) | [onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) and [onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) | [Exporting](tutorials/OnnxTensorflowExport.ipynb) | [Importing](tutorials/OnnxTensorflowImport.ipynb) [experimental] |
| [Apple CoreML](https://developer.apple.com/documentation/coreml) | [onnx/onnx-coreml](https://github.com/onnx/onnx-coreml) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnxmltools) | [Importing](tutorials/OnnxCoremlImport.ipynb) |
| [SciKit-Learn](http://scikit-learn.org/) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnxmltools) | n/a |
| [ML.NET](https://github.com/dotnet/machinelearning/) | [built-in](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.models.onnxconverter.convert?view=ml-dotnet#definition) | [Exporting](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxTests.cs) | [Importing](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.OnnxTransformTest/OnnxTransformTests.cs#L186) |
| [Menoh](https://github.com/pfnet-research/menoh) | [pfnet-research/menoh](https://github.com/pfnet-research/menoh) | n/a | [Importing](tutorials/OnnxMenohHaskellImport.ipynb) |
| [MATLAB](https://www.mathworks.com/) | [onnx converter on matlab central file exchange](https://www.mathworks.com/matlabcentral/fileexchange/67296) | [Exporting](https://www.mathworks.com/help/deeplearning/ref/exportonnxnetwork.html) | [Importing](https://www.mathworks.com/help/deeplearning/ref/importonnxnetwork.html) |
| [TensorRT](https://developer.nvidia.com/tensorrt) | [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) | n/a | [Importing](https://github.com/onnx/onnx-tensorrt/blob/master/README.md) |

* Use services like [Azure Custom Vision service](https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/custom-vision-onnx-windows-ml) that generate customized ONNX models for your data 

## End-to-end tutorials

* For preparation 
  * [Docker image for Caffe2/PyTorch/ONNX tutorials](pytorch_caffe2_docker.md)
  * [Caffe/Keras/CoreML Docker Converter Image](https://hub.docker.com/r/microsoft/onnxconverter/)
  
* For serving
  * [Serving ONNX models with MXNet Model Server](tutorials/ONNXMXNetServer.ipynb)

* For conversion
  * [Convert a PyTorch model to Tensorflow using ONNX](tutorials/PytorchTensorflowMnist.ipynb)
  
* From conversion to deployment
  * [Converting SuperResolution model from PyTorch to Caffe2 with ONNX and deploying on mobile device](tutorials/PytorchCaffe2SuperResolution.ipynb)
  * [Transferring SqueezeNet from PyTorch to Caffe2 with ONNX and to Android app](tutorials/PytorchCaffe2MobileSqueezeNet.ipynb)
  * [Converting Style Transfer model from PyTorch to CoreML with ONNX and deploying to an iPhone](https://github.com/onnx/tutorials/tree/master/examples/CoreML/ONNXLive)
  * [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
  * [MXNet to ONNX to ML.NET with SageMaker, ECS and ECR](https://cosminsanda.com/posts/mxnet-to-onnx-to-ml.net-with-sagemaker-ecs-and-ecr/) - external link
 
## ONNX tools

* [Verifying correctness and comparing performance](tutorials/CorrectnessVerificationAndPerformanceComparison.ipynb)
* [Visualizing an ONNX model](tutorials/VisualizingAModel.md) (useful for debugging)
* [Netron: a viewer for ONNX models](https://github.com/lutzroeder/Netron)
* [Example of operating on ONNX protobuf](https://github.com/onnx/onnx/blob/master/onnx/examples/Protobufs.ipynb)

## Contributing

We welcome improvements to the convertor tools and contributions of new ONNX bindings. Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

Use ONNX for something cool? Send the tutorial to this repo by submitting a PR.
